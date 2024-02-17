#define PORT_CUDA
#include "hiccl.h"

int main() {

  size_t count = 256e6; // 1 GB
  float *weights;
  cudaMalloc(&weights, count * sizeof(float));

  HiCCL::Comm<float> allreduce;
  {
    // END POINTS
    float *sendbuf;
    float *recvbuf;
    CommBench::allocate(sendbuf, count);
    CommBench::allocate(recvbuf, count);

    // COMPOSITION
    int root = 0;
    allreduce.add_reduce(sendbuf, 0, recvbuf, 0, count, HiCCL::all, root); // all -> root
    allreduce.add_fence();
    allreduce.add_bcast(recvbuf, 0, recvbuf, 0, count, root, HiCCL::others); // root -> all - root

    // SET PARAMETERS
    // allreduce.set_hierarchy(std::vector<int> {2, 4}, std::vector<CommBench::library> {CommBench::XCCL, CommBench::IPC_get});
    // allreduce.set_numstripe(4);
    // allreduce.set_pipedepth(12);
    allreduce.set_endpoints(sendbuf, count, recvbuf, count);
    allreduce.init();
    CommBench::report_memory();
  }

  allreduce.run(weights, weights);

  HiCCL::measure(5, 10, count, allreduce);

  cudaFree(weights);

  return 0;
}
