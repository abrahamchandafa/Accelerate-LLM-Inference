/**
 * Some common headers and definitions helpful for your impl:
 * 1. `sched_attr` definition for SYS_sched_getattr and SYS_sched_setattr
 * 2. `get_sched_name` to get scheduling algo name based on macro defn
 * 3. `HANDLE_ERROR` macro helpful to handler error
 * 
 * You SHALL NOT change this file
 */

#pragma once // avoid redefinition

#define _GNU_SOURCE // for glibc
#include <time.h>

#define MAX_PROMPT_LEN 512
#define MAX_NEW_TOKENS 256

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define FILELINE __FILE__ ":" TOSTRING(__LINE__)

// a helper macro to locate error and reason, add to anywhere might raise
// can also be used as breakpoint:
// * if no error, will print "Success"
// * if do error, will print the error info like Segment Fault
#define HANDLE_ERROR() do { \
    perror(FILELINE);       \
    exit(EXIT_FAILURE);     \
} while(0)

// a helper function to measure the timestamp in milliseconds
long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}
