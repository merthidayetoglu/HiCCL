/* Copyright 2023 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h> // for printf
#include <stdlib.h> // for atoi
#include <cstring> // for memcpy
#include <algorithm> // for sort
#include <mpi.h>
#include <omp.h>

#define ROOT 0

// HEADERS
 #include <nccl.h>
// #include <rccl.h>
// #include <sycl.hpp>
// #include <ze_api.h>

// PORTS
 #define PORT_CUDA
// #define PORT_HIP
// #define PORT_SYCL

#include "../CommBench/verification/coll.h"

#include "exacomm.h"

// UTILITIES
#include "../CommBench/util.h"
void print_args();

// USER DEFINED TYPE
#define Type int
/*struct Type
{
  // int tag;
  int data[1];
  // complex<double> x, y, z;
};*/

int main(int argc, char *argv[])
{
  // INITIALIZE
  int myid;
  int numproc;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  int numthread;
  #pragma omp parallel
  if(omp_get_thread_num() == 0)
    numthread = omp_get_num_threads();
  // char machine_name[MPI_MAX_PROCESSOR_NAME];
  // int name_len = 0;
  // MPI_Get_processor_name(machine_name, &name_len);
  // printf("myid %d %s\n",myid, machine_name);

  // INPUT PARAMETERS
  int pattern = atoi(argv[1]);
  int numbatch = atoi(argv[2]);
  size_t count = atol(argv[3]);
  int warmup = atoi(argv[4]);
  int numiter = atoi(argv[5]);

  // PRINT NUMBER OF PROCESSES AND THREADS
  if(myid == ROOT)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Number of threads per proc: %d\n", numthread);
    printf("Number of warmup %d\n", warmup);
    printf("Number of iterations %d\n", numiter);

    printf("Pattern: ");
    switch(pattern) {
      case ExaComm::gather : printf("Gather\n"); break;
      case ExaComm::scatter: printf("Scatter\n"); break;
      case ExaComm::reduce: printf("Reduce\n"); break;
      case ExaComm::broadcast : printf("Broadcast\n"); break;
      case ExaComm::alltoall : printf("All-to-All\n"); break;
      case ExaComm::allgather : printf("All-Gather\n"); break;
      case ExaComm::allreduce : printf("All-Reduce\n"); break;
    }
    printf("Number of batches: %d\n", numbatch);

    printf("Bytes per Type %lu\n", sizeof(Type));
    printf("Point-to-point (P2P) count %ld ( %ld Bytes)\n", count, count * sizeof(Type));
    printf("\n");
  }

  setup_gpu();

  // ALLOCATE
  Type *sendbuf_d;
  Type *recvbuf_d;
#ifdef PORT_CUDA
  cudaMalloc(&sendbuf_d, count * numproc * sizeof(Type));
  cudaMalloc(&recvbuf_d, count * numproc * sizeof(Type));
#elif defined PORT_HIP
  hipMalloc(&sendbuf_d, count * numproc * sizeof(Type));
  hipMalloc(&recvbuf_d, count * numproc * sizeof(Type));
#elif defined PORT_SYCL
  sycl::queue q(sycl::gpu_selector_v);
  sendbuf_d = sycl::malloc_device<Type>(count * numproc, q);
  recvbuf_d = sycl::malloc_device<Type>(count * numproc, q);
#else
  sendbuf_d = new Type[count * numproc];
  recvbuf_d = new Type[count * numproc];
#endif

  {
    ExaComm::printid = myid;
    ExaComm::Comm<Type> bench(MPI_COMM_WORLD);

    switch (pattern) {
      case ExaComm::pt2pt :
        bench.add(sendbuf_d, 0, recvbuf_d, 0, count, 0, 4);
        break;
      case ExaComm::gather :
        for(int p = 0; p < numproc; p++)
          bench.add(sendbuf_d, 0, recvbuf_d, p * count, count, p, ROOT);
        break;
      case ExaComm::scatter :
        for(int p = 0; p < numproc; p++)
          bench.add(sendbuf_d, p * count, recvbuf_d, 0, count, ROOT, p);
        break;
      case ExaComm::broadcast :
      {
        /* for(int p = 0; p < numproc; p++)
          bench.add(sendbuf_d, 0, recvbuf_d, 0, count, ROOT, p); */
        std::vector<int> recvids;
        for(int p = 0 ; p < numproc; p++)
          recvids.push_back(p);
        bench.add(sendbuf_d, 0, recvbuf_d, 0, count, ROOT, recvids);
        break;
      }
      case ExaComm::alltoall :
      {
        for(int sender = 0; sender < numproc; sender++)
          for(int recver = 0; recver < numproc; recver++)
            bench.add(sendbuf_d, recver * count, recvbuf_d, sender * count, count, sender, recver);
        break;
      }
      case ExaComm::allgather :
      {
        for(int sender = 0; sender < numproc; sender++) {
          //for(int recver = 0; recver < numproc; recver++)
          //  bench.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, recver);
          std::vector<int> recvids;
          for(int p = 0 ; p < numproc; p++)
            recvids.push_back(p);
          bench.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, recvids);
	}
        break;
      }
    }

    int numlevel = 4;
    int groupsize[4] = {numproc, 16, 8, 4};
    CommBench::library library[4] = {CommBench::NCCL, CommBench::NCCL, CommBench::NCCL, CommBench::IPC};

    bench.init(numlevel, groupsize, library, numbatch);

    // bench.run_batch();
    // bench.overlap_batch();

    bench.measure(warmup, numiter);
    // bench.report();

    ExaComm::measure(count * numproc, warmup, numiter, bench);
    ExaComm::validate(sendbuf_d, recvbuf_d, count, pattern, bench);
  }

// DEALLOCATE
#ifdef PORT_CUDA
  cudaFree(sendbuf_d);
  cudaFree(recvbuf_d);
#elif defined PORT_HIP
  hipFree(sendbuf_d);
  hipFree(recvbuf_d);
#elif defined PORT_SYCL
  sycl::free(sendbuf_d, q);
  sycl::free(recvbuf_d, q);
#else
  delete[] sendbuf_d;
  delete[] recvbuf_d;
#endif

  // FINALIZE
  MPI_Finalize();

  return 0;
} // main()

