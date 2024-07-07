# HiCCL

HiCCL is a compositional communication library for hierarchical GPU networks. It offers an API for composing collective functions using *multicast*, *reduction*, and *fence* primitives. These primitives are machine- and library-agnostic, and are defined across GPU endpoints. The compositionality decouples the higher-level communication design and machine-specific optimizations. This design is to provide productivity and portability when building custom collective functions.

for separating the collective pattern design from the machine-specific optimizations. HiCCL optimizes the given high-level pattern for a specified machine that is described by the user. The implementation is achieved by using the point-to-point functions of the native communication libraries, such as IPC (put, get), MPI, NCCL, RCCL, and OneCCL.

HiCCL is based on [CommBench](https://github.com/merthidayetoglu/CommBench): a micro-benchmarking software for HPC networks. While HiCCL is a C++ layer for generating communication patterns, CommBench is the middleware for implementing the patterns on an actual machine.

For questions and support, please send an email to merth@stanford.edu

![Collective throughput.](misc/hiccl_collectives_new.png)
