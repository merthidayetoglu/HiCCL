#define PORT_SYCL
#include "hiccl.h"

int main() {

  size_t count = 256e6; // 1 GB
  float *weights;
  // cudaMalloc(&weights, count * sizeof(float));
  CommBench::allocate(weights, count);

  HiCCL::Comm<float> allreduce;
  {
    float *sendbuf;
    float *recvbuf;
    CommBench::allocate(sendbuf, count);
    CommBench::allocate(recvbuf, count);

    int root = 0;
    allreduce.add_reduce(sendbuf, 0, recvbuf, 0, count, HiCCL::all, root); // all -> root
    allreduce.add_fence();
    allreduce.add_bcast(recvbuf, 0, recvbuf, 0, count, root, HiCCL::others); // root -> all - root

    allreduce.set_hierarchy(std::vector<int> {6, 2}, std::vector<CommBench::library> {CommBench::MPI, CommBench::MPI});
    allreduce.set_endpoints(sendbuf, count, recvbuf, count);
    allreduce.init();
    CommBench::report_memory();
  }
  return 0;

  allreduce.run(weights, weights);
  allreduce.measure(5, 10, count);

  // cudaFree(weights);
  CommBench::free(weights);

  return 0;
}
