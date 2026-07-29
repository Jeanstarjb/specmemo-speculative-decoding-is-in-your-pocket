# SpecMemo: Speculative Decoding is in Your Pocket

**Research Paper:** [https://arxiv.org/pdf/2506.01986v1](https://arxiv.org/pdf/2506.01986v1)

## The Mission
The inability to efficiently deploy large language models (LLMs) on memory-constrained devices, such as mobile GPUs or small server GPUs, limits access to advanced AI applications in resource-constrained environments, hindering democratized AI adoption.

## Architecture
The solution involves developing a device-aware inference engine, SpecMemo, which uses speculative decoding optimized for memory-constrained devices. The architecture includes a backend for memory-efficient speculative decoding, a distributed processing layer for multi-GPU setups, and a frontend API for seamless integration. The tech stack includes Python, PyTorch for LLM inference, CUDA for GPU optimization, FastAPI for API development, Docker for containerization, and Kubernetes for multi-GPU orchestration.

## Progress Log

- **Completed Task:** Set up the project repository with basic configurations, including Python environment, dependencies, and Dockerfile for containerization.