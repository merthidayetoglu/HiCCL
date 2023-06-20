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

// UTILITIES
#include "../../CommBench/util.h"

#include "../exacomm.h"
#include "coll.h"

void print_args();

// USER DEFINED TYPE
#define Type int
/*struct Type
{
  // int tag;
  int data[1];
  // complex<double> x, y, z;
};*/

void *thread_function(void *dummyPtr) {
  int *ptr = (int*) dummyPtr;
  printf("hello from %d\n", *ptr);
  // printf("Thread number %ld\n", pthread_self());
  return NULL;
}


int main(int argc, char *argv[])
{
  // INITIALIZE
  int myid;
  int numproc;
  /*int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if(provided != MPI_THREAD_MULTIPLE) {
    printf("provide MPI_THREAD_MULTIPLE!! current provide is %d\n", provided);
    return 0;
  }*/
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  // char machine_name[MPI_MAX_PROCESSOR_NAME];
  // int name_len = 0;
  // MPI_Get_processor_name(machine_name, &name_len);
  // printf("myid %d %s\n",myid, machine_name);

  // INPUT PARAMETERS
  int pattern = atoi(argv[1]);
  int divide = atoi(argv[2]);
  size_t count = atol(argv[3]);
  int warmup = atoi(argv[4]);
  int numiter = atoi(argv[5]);

  // PRINT NUMBER OF PROCESSES AND THREADS
  if(myid == ROOT)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Number of warmup %d\n", warmup);
    printf("Number of iterations %d\n", numiter);

    printf("Pattern: %d\n", pattern);
    printf("Divide: %d\n", divide);

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
    char filename[1024];
    sprintf(filename, "output_%d.txt", ExaComm::printid);
    ExaComm::pFile = fopen(filename, "w");

    int numlevel = 3;
    int groupsize[5] = {numproc, 4, 1, 1, 1};
    CommBench::library library[5] = {CommBench::NCCL, CommBench::MPI, CommBench::IPC, CommBench::IPC, CommBench::IPC};

    int recvid[numproc];
    for(int p = 0; p < numproc; p++)
      recvid[p] = p;

    std::vector<std::list<CommBench::Comm<Type>*>> commlist(divide * numproc);

    for(int p = 0; p < numproc; p++)
      for(int div = 0; div < divide; div++)
      {
        ExaComm::BCAST<Type> bcast(sendbuf_d, div * (count / divide), recvbuf_d, p * count + div * (count / divide), count / divide, p, numproc, recvid);
        ExaComm::bcast_tree(MPI_COMM_WORLD, numlevel, groupsize, library, bcast, commlist[p * divide + div], 1);
      }

    // omp_set_num_threads(1);

    for(int p = 0; p < commlist.size(); p++) {
      for(auto comm : commlist[p]) {
        comm->report();
        // comm->run();
        // comm->measure(warmup, numiter);
      }
      if(myid == ROOT)
        printf("commlist[%d] size %zu\n", p, commlist[p].size());
    }

    measure(count * numproc, warmup, numiter, commlist);

    // ExaComm::run_concurrent(commlist);

    validate(sendbuf_d, recvbuf_d, count, pattern, commlist);


    /*for(int sender = 0; sender < numproc; sender++) {
      ExaComm::BCAST<Type> bcast(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, numproc, recvid);
      ExaComm::bcast_tree(MPI_COMM_WORLD, frontier_numlevel, frontier_groupsize, frontier_library, bcast, commlist[sender], 1);
    }

    for(int p = 0; p < numproc; p++) {
      for(auto comm : commlist[p])
        // comm->measure(warmup, numiter);
        comm->run();
      if(myid == ROOT)
        printf("commlist[%d]\n", p);
    }

    return 0;

    
    if(ExaComm::printid == ROOT)
      printf("commlist size %lu\n", commlist[0].size());

    std::list<CommBench::Comm<Type>*> commlist_all;
    for(int p = 0; p < numproc; p++)
      for(auto comm : commlist[p])
        commlist_all.push_back(comm);

    measure(count * numproc, warmup, numiter, commlist_all);
    validate(sendbuf_d, recvbuf_d, count, pattern, commlist_all);*/
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

void print_args() {

  int myid;
  int numproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  if(myid == ROOT) {
    printf("\n");
    printf("CommBench requires nine arguments:\n");
    printf("1. library: 0 for IPC, 1 for MPI, 2 for NCCL or RCCL\n");
    printf("2. pattern: 1 for Rail, 2 for Dense, 3 for Fan\n");
    printf("3. direction: 1 for unidirectional, 2 for bidirectional, 3 for omnidirectional\n");
    printf("4. count: number of 4-byte elements\n");
    printf("5. warmup: number of warmup rounds\n");
    printf("6. numiter: number of measurement rounds\n");
    printf("7. p: number of processors\n");
    printf("8. g: group size\n");
    printf("9. k: subgroup size\n");
    printf("where on can run CommBench as\n");
    printf("mpirun ./CommBench library pattern direction count warmup numiter p g k\n");
    printf("\n");
  }
}

