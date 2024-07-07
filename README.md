# HiCCL

HiCCL is a compositional communication library for hierarchical GPU networks. It offers an API for composing collective functions using *multicast*, *reduction*, and *fence* primitives. These primitives are machine- and library-agnostic, and are defined across GPU endpoints. HiCCL's design principle is to decouple the higher-level communication design and machine-specific optimizations. This principle aims to improve productivity, portability, and performance when building custom collective functions.


HiCCL is based on [CommBench](https://github.com/merthidayetoglu/CommBench): a micro-benchmarking software for HPC networks. While HiCCL is a C++ layer for generating communication patterns on an abstract machine, CommBench is the middleware for implementing the patterns on an actual machine. The implementation is achieved by using the point-to-point functions of the chosen communication library, MPI, NCCL, RCCL, and OneCCL, and IPC capability (put, get), and recently GASNet-EX for non-MPI applications.

## API

The collective function is built within a persistent communicator. As an example, below shows an in-place composition of the All-Reduce collective.

```c++
#define PORT_CUDA
#include "hiccl.h"

#define T float;

int main() {

  size_t numelements = 1e9 / sizeof(T); // 1 GB

  HiCCL::Comm<T> allreduce;

  T *sendbuf;
  T *recvbuf;
  CommBench::allocate(sendbuf, numelements);
  CommBench::allocate(recvbuf, numelements);

  // reduce-scatter
  for (int i = 0; i < CommBench::numproc; i++)
    allreduce.add_reduction(sendbuf + i * numelements / CommBench::numproc, recvbuf, numelements, HiCCL::all, i);
  // express ordering
  allreduce.add_fence();
  // all-gather
  for (int i = 0; i < CommBench::numproc; i++)
    allreduce.add_multicast(temp, buffer, numelements, i, HiCCL::all);

}

```

![Collective throughput.](misc/hiccl_collectives_new.png)

For questions and support, please send an email to merth@stanford.edu
