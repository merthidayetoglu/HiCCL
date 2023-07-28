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
#define Type float
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

  enum pattern {pt2pt, scatter, gather, broadcast, reduce, alltoall, allgather, allreduce, reducescatter};

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
      case gather    : printf("Gather\n");     break;
      case scatter   : printf("Scatter\n");    break;
      case reduce    : printf("Reduce\n");     break;
      case broadcast : printf("Broadcast\n");  break;
      case alltoall  : printf("All-to-All\n"); break;
      case allgather : printf("All-Gather\n"); break;
      case allreduce : printf("All-Reduce\n"); break;
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

  std::vector<int> proclist;
  for(int p = 0 ; p < numproc; p++)
    proclist.push_back(p);
  std::vector<std::vector<int>> recvids(numproc);
  for(int sender = 0; sender < numproc; sender++)
    for(int recver = 0; recver < numproc; recver++)
      if(recver != sender)
        recvids[sender].push_back(recver);
  size_t count_part = count / numproc;


  // PATTERN DESRIPTION
  {
    ExaComm::printid = myid;
    ExaComm::Comm<Type> coll(MPI_COMM_WORLD);

    switch (pattern) {
      case scatter :
        for(int p = 0; p < numproc; p++)
          coll.add_reduce(sendbuf_d, p * count, recvbuf_d, 0, count, ROOT, p);
        break;
      case gather :
        for(int p = 0; p < numproc; p++)
          coll.add_bcast(sendbuf_d, 0, recvbuf_d, p * count, count, p, ROOT);
        break;
      case broadcast :
        // coll.add(sendbuf_d, 0, recvbuf_d, 0, count, ROOT, proclist);
        for(int recver = 0; recver < numproc; recver++)
          coll.add_reduce(sendbuf_d, recver * count_part, recvbuf_d, recver * count_part, count_part, ROOT, recver);
        coll.fence();
        for(int sender = 0; sender < numproc; sender++)
          coll.add_bcast(recvbuf_d, sender * count_part, recvbuf_d, sender * count_part, count_part, sender, recvids[sender]);
        break;
      case reduce :
        // coll.add_reduce(sendbuf_d, 0, recvbuf_d, 0, count, proclist, ROOT);
	for(int recver = 0; recver < numproc; recver++)
          coll.add_reduce(sendbuf_d, recver * count_part, recvbuf_d, recver * count_part, count_part, proclist, recver);
        coll.fence();
        for(int sender = 0; sender < numproc; sender++)
          coll.add_bcast(recvbuf_d, sender * count_part, recvbuf_d, sender * count_part, count_part, sender, ROOT);
        break;
      case alltoall :
        for(int sender = 0; sender < numproc; sender++)
          for(int recver = 0; recver < numproc; recver++)
            coll.add_bcast(sendbuf_d, recver * count, recvbuf_d, sender * count, count, sender, recver);
        break;
      case allgather :
        for(int sender = 0; sender < numproc; sender++)
          coll.add_bcast(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, proclist);
        break;
      case allreduce :
        // for(int recver = 0; recver < numproc; recver++)
        //   coll.add(sendbuf_d, 0, recvbuf_d, 0, count, proclist, recver);
        for(int recver = 0; recver < numproc; recver++)
          coll.add_reduce(sendbuf_d, recver * count_part, recvbuf_d, recver * count_part, count_part, proclist, recver);
        coll.fence();
        for(int sender = 0; sender < numproc; sender++)
          coll.add_bcast(recvbuf_d, sender * count_part, recvbuf_d, sender * count_part, count_part, sender, recvids[sender]);
        break;
      case reducescatter :
        for(int recver = 0; recver < numproc; recver++)
          coll.add_reduce(sendbuf_d, recver * count, recvbuf_d, 0, count, proclist, recver);
        break;
      default:
        if(myid == ROOT)
          printf("invalid collective option\n");
    }

    // MACHINE DESCRIPTION
    int numlevel = 2;
    int groupsize[6] = {numproc, 4, 4, 4, 2, 1};
    CommBench::library library[6] = {CommBench::NCCL, CommBench::IPC, CommBench::IPC, CommBench::IPC, CommBench::IPC, CommBench::IPC};

    double time = MPI_Wtime();
    coll.init(numlevel, groupsize, library, numbatch);
    time = MPI_Wtime() - time;
    if(myid == ROOT)
      printf("preproc time: %e\n", time);

    coll.measure(warmup, numiter);
    // coll.report();

    ExaComm::measure(count * numproc, warmup, numiter, coll);
    ExaComm::validate(sendbuf_d, recvbuf_d, count, pattern, coll);
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

