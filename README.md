# HiCCL

HiCCL is a hierarchical communication library. It has a compositional API for separating the collective pattern design from the machine-specific optimizations. HiCCL optimizes the given high-level pattern for a specified machine that is described by the user. The implementation is achieved by using the point-to-point functions of the native communication libraries, such as IPC (put, get), MPI, NCCL, RCCL, and OneCCL.

HiCCL is based on [CommBench](https://github.com/merthidayetoglu/CommBench): a micro-benchmarking software for HPC networks. While HiCCL is a C++ layer for generating communication patterns, CommBench is the middleware for implementing the patterns on an actual machine.

For questions and support, please send an email to merth@stanford.edu
