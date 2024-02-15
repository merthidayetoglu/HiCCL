#define PORT_CUDA
#include "hiccl.h"

int main() {

  size_t count = 256e6; // 1 GB
  float *weights;
  cudaMalloc(&weights, count * sizeof(float));

  HiCCL::Comm<float> allreduce;
  {
    float *sendbuf;
    float *recvbuf;
    CommBench::allocate(sendbuf, count);
    CommBench::allocate(recvbuf, count);

    int root = 0;
    allreduce.add_reduce(sendbuf, 0, recvbuf, 0, count, CommBench::numproc, root); // all -> root
    allreduce.add_fence();
    allreduce.add_bcast(recvbuf, 0, recvbuf, 0, count, root, -1); // root -> all - root

    allreduce.init();
    allreduce.set_endpoints(sendbuf, count, recvbuf, count);
  }

  allreduce.run(weights, weights);
  allreduce.measure(5, 10, count);

  cudaFree(weights);

  return 0;
}
