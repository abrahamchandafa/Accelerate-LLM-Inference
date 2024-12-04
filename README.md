# Accelerate-LLM-Inference

Implemented a multi-threaded version of Llama3, an open-source GPT variant, to accelerate inference tasks, focusing on matrix-vector multiplication and multi-head attention computations.

Utilized POSIX Pthreads and synchronization mechanisms (semaphores/mutexes) to manage and coordinate threads in a shared memory environment.
Designed a thread pool to optimize performance by reusing threads for multiple inference tasks, significantly enhancing the throughput of the language model.

Conducted performance benchmarking, demonstrating improved speed (tokens per second) compared to a single-threaded implementation while ensuring identical output consistency.

# Tech stack:
- C Programming Language:<br/>
The primary language used for implementing the inference algorithms and multi-threading logic.
- POSIX Pthreads:<br/>
A standard for multi-threading in C, allowing for the creation, management, and synchronization of threads.
- Semaphores and Mutexes:<br/>
Synchronization mechanisms used to manage access to shared resources and ensure thread safety.
- GNU Make:<br/>
A build automation tool used to compile the C code and manage dependencies.
- Llama3 Model:<br/>
An open-source variant of GPT utilized for the inference tasks.
- Linux Environment:<br/>
The development and testing were conducted on a Linux platform, ensuring compatibility with course requirements.
