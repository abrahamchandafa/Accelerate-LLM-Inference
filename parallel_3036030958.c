
#include "common.h" // some common definitions

#include <unistd.h>       // for nearly everything :)
#include <stdio.h>        // for printf, sprintf, fgets
#include <stdlib.h>       // for malloc, calloc
#include <stdint.h>       // for uint8_t and uint64_t
#include <time.h>         // for time
#include <string.h>       // for memcpy and strcmp
#include <sys/resource.h> // for rusage collection

#include "model.h"// for Llama definitions -> no need to know

int pos = 0; // global position of generation
Transformer transformer; // transformer instance to be init
Tokenizer tokenizer;     // tokenizer instance to be init
Sampler sampler;         // sampler instance to be init

// YOUR CODE STARTS HERE
#include <pthread.h>
// #include <semaphore.h> // uncomment this line if you use semaphore
#include <stdbool.h>   // uncomment this line if you want true / false

// you may define global variables here

pthread_t* threads; // later malloc for threads
int* ids; // pass and store pseudo-ids for threads i.e. 0, 1, ... n-1
struct rusage *collector; // for collecting usage stats 
int* do_mvm; // 0s (done) and 1s (not done) for each thread working on mvm
int* do_mha; // 0s (done) and 1s (not done) for each thread working on mha

bool terminate = false; 

int thr_count; // total number of threads 
int workload; // either n_heads or rows 

pthread_cond_t one = PTHREAD_COND_INITIALIZER; // for main thread to wait on threads working
pthread_mutex_t one_lock = PTHREAD_MUTEX_INITIALIZER; // associated lock

pthread_cond_t cond = PTHREAD_COND_INITIALIZER; // for threads to wait on job assignment
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER; // associated lock

typedef struct {
    float* out;
    QuantizedTensor *vec; 
    QuantizedTensor *mat; 
    int col;
    int row;
} mvm_stuff; // mvm variables for global access...
mvm_stuff mvm_data;

typedef struct {
    float* out;
    float* q; 
    float* key_cache;
    float* value_cache;
    float* att;
    int seq_len;
    int n_heads;
    int head_size;
    int kv_dim;
    int kv_mul;
} mha_stuff; // mha variables for global access...
mha_stuff mha_data;

// function executed by each thread to complete mat_vec_mul
// @note: please modify the signature to what you want
void mat_vec_mul_task_func(int id) {

    //my logic: assigning every thread equal jobs, but break if out of boundary
    // this will automatically handle any end threads
    int count;
    int start;
    int end;
    if (mvm_data.row%thr_count == 0){ 
        count = (int) mvm_data.row/thr_count;
    } else {
        count = (int) (mvm_data.row/thr_count) + 1;
    }
    start = id * count;
    end = start + count;

    for (int i = start; i < end; i++) {
        if (i >= mvm_data.row)
            break;

        float val = 0.0f; // final value
        int32_t ival = 0; // integer value to be dequantized
        int in = i * mvm_data.col;   // 

        // for each column
        // GS is group size of quantization, not included in assignment
        // @note please don't parallel this loop
        for (int j = 0; j <= mvm_data.col - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) mvm_data.vec->q[j + k]) * ((int32_t) mvm_data.mat->q[in + j + k]);
            }
            val += ((float) ival) * mvm_data.mat->s[(in + j) / GS] * mvm_data.vec->s[j / GS];
            ival = 0;
        }
        mvm_data.out[i] = val;
    }
}

// function executed by each thread to complete multi_head_attn
// @note: please modify the signature to what you want
void multi_head_attn_task_func(int id) {

    //my logic: assigning every thread equal jobs, but break if out of boundary
    // this will automatically handle any end threads
    int count = 0;
    int start = 0;
    int end = 0;

    if (mha_data.n_heads % thr_count == 0) { 
        count = mha_data.n_heads / thr_count;
    } else { 
        count = (int) (mha_data.n_heads / thr_count) + 1;
    }
    start = id * count;
    end = start + count;

    for (int h = start; h < end; h++) {
        if (h >= mha_data.n_heads)
            break;

        // Get the query vector for this head
        float* head_q = mha_data.q + h * mha_data.head_size;
        // Attention scores for this head
        float* head_att = mha_data.att + h * mha_data.seq_len;

        // Iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // Get the key vector for this head and at this timestep
            float* head_k = mha_data.key_cache + t * mha_data.kv_dim + (h / mha_data.kv_mul) * mha_data.head_size;

            // Calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < mha_data.head_size; i++) {
                score += head_q[i] * head_k[i];
            }
            score /= sqrtf(mha_data.head_size);

            // Save the score to the attention buffer
            head_att[t] = score;
        }

        // Softmax the scores to get attention weights, from 0..pos inclusively
        softmax(head_att, pos + 1);

        // Weighted sum of the values, store back into out
        float* head_out = mha_data.out + h * mha_data.head_size;
        memset(head_out, 0, mha_data.head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            // Get the value vector for this head and at this timestep
            float* head_v = mha_data.value_cache + t * mha_data.kv_dim + (h / mha_data.kv_mul) * mha_data.head_size;

            // Get the attention weight for this timestep
            float a = head_att[t];

            // Accumulate the weighted value into head out
            for (int i = 0; i < mha_data.head_size; i++) {
                head_out[i] += a * head_v[i];
            }
        }
    }

}

// thread function used in pthread_create
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void *thr_func(void *arg) {
    int id = *(int *) arg;
    while(1) {
        pthread_mutex_lock(&mutex);
        while(!do_mvm[id] && !do_mha[id] && !terminate){ // each thread checks if theres a task, and no terminate
            pthread_cond_wait(&cond, &mutex);
        }        
        pthread_mutex_unlock(&mutex);

        if (terminate){ // termination
            getrusage(RUSAGE_THREAD, &collector[id]);
            pthread_exit(NULL);
        }
        else if (do_mvm[id]) { // an mvm task for a given thread
            // my logic: after each thread does the task, it reduces the workload
            mat_vec_mul_task_func(id);

            pthread_mutex_lock(&mutex); 
            do_mvm[id] = 0; // thread marks itself as done to avoid immediate unwanted re-calling 
            pthread_mutex_unlock(&mutex);

            pthread_mutex_lock(&one_lock); 
            workload--;   // workload deduction
            pthread_cond_signal(&one); // inform main thread, which will check if all are done
            pthread_mutex_unlock(&one_lock);
        } 
        else if (do_mha[id]) { // an mha task for a given thread
            // my logic: after each thread does the task, it reduces the workload
            multi_head_attn_task_func(id);

            pthread_mutex_lock(&mutex);
            do_mha[id] = 0; // thread marks itself as done to avoid unwanted re-calling
            pthread_mutex_unlock(&mutex);

            pthread_mutex_lock(&one_lock);
            workload--;         // workload deduction
            pthread_cond_signal(&one); // inform main thread, which will check if all are done
            pthread_mutex_unlock(&one_lock);
        }
    }
    return NULL;
}

// function to initialize thread pool
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void init_thr_pool(int num_thr) {

    thr_count = num_thr; // global thread number
    workload = num_thr; // this will be keeping track of tasking and waiting

    threads = malloc(num_thr * sizeof(pthread_t));
    ids = malloc(num_thr * sizeof(int));
    collector = malloc(num_thr * sizeof(struct rusage));
    do_mvm = malloc(num_thr * sizeof(int));
    do_mha = malloc(num_thr * sizeof(int));

    for (int i = 0; i < num_thr; i++) {
        do_mha[i] = 0; // each thread has no task initially
        do_mvm[i] = 0; // each thread has no task initially
    }
    
    for (int i = 0; i < num_thr; i++) { // create threads, and pass their custom ids to thr_func
        ids[i] = i;
        pthread_create(&threads[i], NULL, thr_func, &ids[i]);
    }    
}

// function to close thread pool
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void close_thr_pool() {
    //logic: aquire lock, set terminate to true, and wake threads up
    // wait for threads to exit (pthread_join) then
    // print their usages which are in collector

    pthread_mutex_lock(&mutex);
    terminate = true;
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&mutex);
    
    for (int i = 0; i < thr_count; i++) {
        pthread_join(threads[i], NULL);       
    }

    for (int i = 0; i < thr_count; i++){
        printf("\033[0;32mThread %d has completed - user: %.4f s, system: %.4f s \033[0m\n",i, 
        (collector[i].ru_utime.tv_sec + collector[i].ru_utime.tv_usec/1000000.0),
        (collector[i].ru_stime.tv_sec + collector[i].ru_stime.tv_usec/1000000.0));
    }

    struct rusage main_thr;
    getrusage(RUSAGE_THREAD, &main_thr);
    printf("\033[0;32mmain thread - user: %.4f s, system: %.4f s \033[0m\n", 
    (main_thr.ru_utime.tv_sec + main_thr.ru_utime.tv_usec/1000000.0),
    (main_thr.ru_stime.tv_sec + main_thr.ru_stime.tv_usec/1000000.0));

    struct rusage w_p;
    getrusage(RUSAGE_SELF, &w_p);
    printf("\033[0;32mWhole process - user: %.4f s, system: %.4f s \033[0m\n", 
    (w_p.ru_utime.tv_sec + w_p.ru_utime.tv_usec/1000000.0),
    (w_p.ru_stime.tv_sec + w_p.ru_stime.tv_usec/1000000.0));

    // clean-ups
    free(threads);
    free(ids);
    free(collector);
    free(do_mvm);
    free(do_mha);
    pthread_mutex_destroy(&mutex);
    pthread_mutex_destroy(&one_lock);
    pthread_cond_destroy(&cond);
    pthread_cond_destroy(&one);

}

// ----------------------------------------------------------------------------
// entry function for multi-threading matrix multiplication
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void mat_vec_mul(float* out, QuantizedTensor *vec, QuantizedTensor *mat, int col, int row) {
// set the global variables
    mvm_data.out = out;
    mvm_data.vec = vec;
    mvm_data.mat = mat;
    mvm_data.col = col;
    mvm_data.row = row;

// assign task to ALL THREADS, then wake ALL THREADS UP 
    pthread_mutex_lock(&mutex);
    for (int i = 0; i < thr_count; i++){
        do_mvm[i] = 1;
    }
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&mutex);

// wait for all threads to finish (each thread reduces workload, till it is 0)
    pthread_mutex_lock(&one_lock);
    while (workload > 0){
        pthread_cond_wait(&one, &one_lock);
    }
    workload = thr_count; // update workload to match total thread count
    pthread_mutex_unlock(&one_lock);

}

// ----------------------------------------------------------------------------
// entry function for multi-threading multi-head-attention
// @note: YOU CAN NOT MODIFY FUNCTION SIGNATURE!!!
void multi_head_attn(
    float* out,         // output tensor [head, head_size]
    float* q,           // query tensor  [head, head_size]
    float* key_cache,   // cache of history key tensor   [kv_head, seq_len, head_size]
    float* value_cache, // cache of history value tensor [kv_head, seq_len, head_size]
    float* att,         // buffer for attention score [head, seq_len]
    int seq_len,
    int n_heads,
    int head_size,
    int kv_dim,
    int kv_mul) {

// set the global variables
    mha_data.out = out;
    mha_data.q = q;
    mha_data.key_cache = key_cache;
    mha_data.value_cache = value_cache;
    mha_data.att = att;
    mha_data.seq_len = seq_len,
    mha_data.n_heads = n_heads;
    mha_data.head_size = head_size;
    mha_data.kv_dim = kv_dim;
    mha_data.kv_mul = kv_mul;

// assign task to all threads, and wake all threads up
    pthread_mutex_lock(&mutex);
    for (int i = 0; i < thr_count; i++){
        do_mha[i] = 1;
    }
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&mutex);

// wait for all threads to finish
    pthread_mutex_lock(&one_lock);
    while (workload > 0){
        pthread_cond_wait(&one, &one_lock);
    }
    workload = thr_count;
    pthread_mutex_unlock(&one_lock);
    

}
// YOUR CODE ENDS HERE

// ----------------------------------------------------------------------------
// forward Transformer, you're not allowed to modify this part
float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->q, &s->xq, w->wq + l, dim, dim);
        mat_vec_mul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        mat_vec_mul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        multi_head_attn(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, 
            p->seq_len, p->n_heads, head_size, kv_dim, kv_mul);

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        mat_vec_mul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        mat_vec_mul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    mat_vec_mul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// generation loop, you're not allowed to modify this part
void generate(char *prompt) {
    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+6) * sizeof(int)); // +6 reserved for prompt template
    encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;        // place holder for next token
    int token = prompt_tokens[0]; // place holder of prev token, kickoff as prompt_tokens[0]
    int end_pos = pos + MAX_NEW_TOKENS + num_prompt_tokens;
    int start_pos = pos;
    long start_time = 0; // to be lazy iniialzied
    while (pos < end_pos) {

        // forward the transformer to get logits for the next token
        float* logits = forward(&transformer, token, pos);

        if (pos < start_pos + num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos - start_pos + 1];
        } else if (pos == end_pos - 2) {
            // reaching the end, force it to close by <|im_end|>
            next = 2; // := <|im_end|>
        } else {
            // otherwise sample the next token from the logits
            next = sample(&sampler, logits);
        }

        pos++;

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(&tokenizer, token, next);
        if (pos >= num_prompt_tokens) {
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }

        token = next;

        // init the timer here because the first iteration can be slower
        if (start_time == 0) { start_time = time_in_ms(); }
    }
    printf("\n");

    long end_time = time_in_ms();
    // \033[0;32m set color to green and \033[0m reset to default, they won't be generate by LLM
    fprintf(stdout, "\033[0;32mlength: %d, speed (tok/s): %.4f \033[0m\n", 
        pos, (pos - start_pos) / (float) (end_time - start_time) * 1000);
    
    free(prompt_tokens);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *model_path     = "model.bin";  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature    = 0.6f;  // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp           = 0.9f;  // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    char *prompt         = NULL;  // prompt strings
    int num_prompt       = 0; // number of prompts
    uint64_t rng_seed    = 0; // seed rng with time by default
    int num_thr          = 0;

    if (argc == 4) {
        num_thr  = atoi(argv[1]);
        rng_seed = atoi(argv[2]);
        prompt   = argv[3];
    } else {
        fprintf(stderr, "Usage:   ./seq <num_thr> <seed> <prompt>\n");
        fprintf(stderr, "Example: ./seq 4 42 \"What is Fibonacci Number?\"\n");
        fprintf(stderr, "Note:    <prompt> must be quoted with \"\", only one prompt supported\n");
        exit(1);
    }

    // parameter validation/overrides
    if (num_thr <= 0 || num_thr > 16) { 
        fprintf(stderr, "num_thr must between 1 and 16 \n");
        exit(EXIT_FAILURE);
    }
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);

    // build the Transformer via the model .bin file
    build_transformer(&transformer, model_path);
    // build the Tokenizer via the tokenizer .bin file
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    // build the Sampler
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // initialize thread pool
    init_thr_pool(num_thr);

    printf("user: %s \n", prompt);
    // perform multi-threading generation
    generate(prompt);
    
    // close thread pool
    close_thr_pool();

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
