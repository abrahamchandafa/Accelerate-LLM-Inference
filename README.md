# Accelerate-LLM-Inference

A high-performance, multi-threaded implementation of Llama3 (open-source GPT variant) designed to accelerate large language model inference through parallel computing.

## Overview

This project implements parallel processing for the most compute-intensive operations in transformer inference:
- **Matrix-vector multiplication** - parallelized across multiple threads
- **Multi-head attention computations** - distributed workload for attention mechanisms
- **Thread pool architecture** - efficient thread reuse and management

## Key Features

- 🚀 **Performance Optimization**: Significantly improved tokens-per-second throughput vs single-threaded implementation
- 🧵 **Multi-threading**: POSIX Pthreads with advanced synchronization (mutexes, condition variables)
- 📊 **Benchmarking**: Built-in performance monitoring and resource usage collection
- 🔒 **Thread Safety**: Robust synchronization mechanisms for shared memory access
- ⚡ **INT8 Quantization**: Memory-efficient model representation with quantized weights
- 🎯 **Identical Output**: Maintains exact consistency with single-threaded reference implementation

## Tech Stack

- **C Programming Language** - Core implementation with optimized algorithms and multi-threading logic
- **POSIX Pthreads** - Industry-standard multi-threading library for thread creation and management  
- **Synchronization Primitives** - Mutexes and condition variables for thread-safe resource access
- **GNU Make** - Build automation and dependency management
- **Llama3 Model** - Open-source GPT variant with INT8 quantization support
- **Linux Environment** - Development and testing platform ensuring POSIX compatibility

## Usage

```bash
# Build the project
make parallel

# Run with custom thread count, seed, and prompt
./parallel <num_threads> <seed> "<prompt>"

# Example
./parallel 4 42 "What is the Fibonacci sequence?"
```

## Architecture

The implementation uses a **thread pool pattern** where:
- Worker threads are pre-allocated and reused across inference tasks
- Matrix-vector operations are divided among available threads
- Multi-head attention computations are parallelized by attention heads
- Synchronization ensures correct execution order and data consistency
