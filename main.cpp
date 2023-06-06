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
// #include <nccl.h>
 #include <rccl.h>
// #include <sycl.hpp>
// #include <ze_api.h>

// PORTS
// #define PORT_CUDA
 #define PORT_HIP
// #define PORT_SYCL

#include "CommBench/comm.h"
#include "CommBench/verification/coll.h"

#include "exacomm.h"

// UTILITIES
#include "CommBench/util.h"
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
  int library = atoi(argv[1]);
  int pattern = atoi(argv[2]);
  int optimization = atoi(argv[3]);
  size_t count = atol(argv[4]);
  int warmup = atoi(argv[5]);
  int numiter = atoi(argv[6]);

  // PRINT NUMBER OF PROCESSES AND THREADS
  if(myid == ROOT)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Number of threads per proc: %d\n", numthread);
    printf("Number of warmup %d\n", warmup);
    printf("Number of iterations %d\n", numiter);

    printf("Library: %d\n", library);
    printf("Pattern: %d\n", pattern);
    printf("Optimization: %d\n", optimization);

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
    ExaComm::Comm<Type> bench(MPI_COMM_WORLD, (CommBench::library) library);

    switch (pattern) {
      case 0:
        bench.add(sendbuf_d, 0, recvbuf_d, 0, count, 0, 4);
        break;
      case 1:
        for(int p = 0; p < numproc; p++)
          bench.add(sendbuf_d, 0, recvbuf_d, p * count, count, p, ROOT);
        break;
      case 2:
        for(int p = 0; p < numproc; p++)
          bench.add(sendbuf_d, p * count, recvbuf_d, 0, count, ROOT, p);
        break;
      case 4: {
        //for(int p = 0; p < numproc; p++)
        //  bench.add(sendbuf_d, 0, recvbuf_d, 0, count, ROOT, p);
	int recvid[numproc];
        for(int p = 0 ; p < numproc; p++)
          recvid[p] = p;
        bench.add(sendbuf_d, 0, recvbuf_d, 0, count, ROOT, numproc, recvid);
        break;
      }
      case 5:
        for(int sender = 0; sender < numproc; sender++)
          for(int recver = 0; recver < numproc; recver++)
            bench.add(sendbuf_d, recver * count, recvbuf_d, sender * count, count, sender, recver);
        break;
      case 7:
        for(int sender = 0; sender < numproc; sender++) {
          // for(int recver = 0; recver < numproc; recver++)
          //  bench.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, recver);
          int recvid[numproc];    
          for(int p = 0 ; p < numproc; p++)
            recvid[p] = p;
          bench.add(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, numproc, recvid);
	}
        break;
    }

    if(optimization == 0)
      bench.init_flat();
    if(optimization == 1)
      bench.init_mixed(4, CommBench::IPC);
    // if(optimization == 2)
      // bench.init_striped(4, CommBench::IPC);
    if(optimization == 3)
      bench.init_bcast(8, CommBench::IPC);

    bench.measure(warmup, numiter);

    measure(count * numproc, warmup, numiter, bench);

    validate(sendbuf_d, recvbuf_d, count, pattern, bench);
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

