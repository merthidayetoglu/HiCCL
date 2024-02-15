#define PORT_CUDA
#include "hiccl.h"

int main() {

  // ALLREDUCE PARAMETERS
  size_t count = 256e6; // 1 GB
  std::vector<int> hierarchy = {2, 4};
  std::vector<CommBench::library> lib = {CommBench::MPI, CommBench::IPC};
  int pipeline(8);

  // INITIALIZE ALLREDUCE
  HiCCL::Comm<float> allreduce = HiCCL::init_allreduce<float>(count, hierarchy, lib, pipeline);

  // ALLOCATE BUFFER
  float *buffer;
  CommBench::allocate(buffer, count);

  return 0;

  // RUN COMMUNICATIONS
  cudaDeviceSynchronize();
  allreduce.run(buffer, buffer);

  // DEALLOCATE BUFFER
  CommBench::free(buffer);

  return 0;
}
