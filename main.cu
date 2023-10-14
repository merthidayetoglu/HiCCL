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

#include <mpi.h>

#define PORT_CUDA
#include "exacomm.h"

#define ROOT 0

// UTILITIES
#include "CommBench/util.h"
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
  // char machine_name[MPI_MAX_PROCESSOR_NAME];
  // int name_len = 0;
  // MPI_Get_processor_name(machine_name, &name_len);
  // printf("myid %d %s\n",myid, machine_name);

  // INPUT PARAMETERS
  int pattern = atoi(argv[1]);
  int numgroup = atoi(argv[2]);
  int numstripe = atoi(argv[3]);
  int stripeoffset = atoi(argv[4]);
  int pipedepth = atoi(argv[5]);
  int pipeoffset = atoi(argv[6]);
  size_t count = atol(argv[7]);
  int warmup = atoi(argv[8]);
  int numiter = atoi(argv[9]);

  enum pattern {pt2pt, gather, scatter, broadcast, reduce, alltoall, allgather, reducescatter, allreduce};

  // PRINT NUMBER OF PROCESSES AND THREADS
  if(myid == ROOT)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Number of warmup %d\n", warmup);
    printf("Number of iterations %d\n", numiter);
    printf("\n");

    printf("Pattern: ");
    switch(pattern) {
      case gather        : printf("Gather\n");         break;
      case scatter       : printf("Scatter\n");        break;
      case broadcast     : printf("Broadcast\n");      break;
      case reduce        : printf("Reduce\n");         break;
      case alltoall      : printf("All-to-All\n");     break;
      case allgather     : printf("All-Gather\n");     break;
      case reducescatter : printf("Reduce-Scatter\n"); break;
      case allreduce     : printf("All-Reduce\n");     break;
    }
    printf("Number of ring groups: %d\n", numgroup);
    printf("Number of stripes: %d\n", numstripe);
    printf("Stripe offset: %d\n", stripeoffset);
    printf("Pipeline depth: %d\n", pipedepth);
    printf("Pipeline offset: %d\n", pipeoffset);

    printf("Bytes per Type %lu\n", sizeof(Type));
    printf("Point-to-point (P2P) count %ld ( %ld Bytes)\n", count, count * sizeof(Type));
    {
      size_t data = 2 * count * numproc * sizeof(Type);
      printf("sendbuf + recvbuf count: %zu + %zu = ", count * numproc, count * numproc);
      if (data < 1e3)
        printf("%zu bytes\n", data);
      else if (data < 1e6)
        printf("%.4f KB\n", data / 1.e3);
      else if (data < 1e9)
        printf("%.4f MB\n", data / 1.e6);
      else if (data < 1e12)
        printf("%.4f GB\n", data / 1.e9);
      else
        printf("data: %.4f TB\n", data / 1.e12);
      printf("\n");
    }
  }

  setup_gpu();

  // ALLOCATE
  Type *sendbuf_d;
  Type *recvbuf_d;
  ExaComm::allocate(sendbuf_d, count * numproc);
  ExaComm::allocate(recvbuf_d, count * numproc);

  // AUXILLIARY DATA STRUCTURES
  std::vector<int> proclist;
  for(int p = 0 ; p < numproc; p++)
    proclist.push_back(p);
  std::vector<std::vector<int>> recvids(numproc);
  for(int sender = 0; sender < numproc; sender++)
    for(int recver = 0; recver < numproc; recver++)
      if(recver != sender)
        recvids[sender].push_back(recver);

  // COLLECTIVE COMMUNICATION
  {
    ExaComm::printid = ROOT;
    ExaComm::Comm<Type> coll;

    // PATTERN DESRIPTION
    switch (pattern) {
      case gather :
        for(int sender = 0; sender < numproc; sender++)
          coll.add_bcast(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, ROOT);
        break;
      case scatter :
        for(int recver = 0; recver < numproc; recver++)
          coll.add_reduce(sendbuf_d, recver * count, recvbuf_d, 0, count, ROOT, recver);
        break;
      case broadcast :
        coll.add_bcast(sendbuf_d, 0, recvbuf_d, 0, count * numproc, ROOT, proclist);
        // SCATTER + ALL-GATHER
        /* for(int recver = 0; recver < numproc; recver++)
          coll.add_reduce(sendbuf_d, recver * count, recvbuf_d, recver * count, count, ROOT, recver);
        coll.add_fence();
        for(int sender = 0; sender < numproc; sender++)
          coll.add_bcast(recvbuf_d, sender * count, recvbuf_d, sender * count, count, sender, recvids[sender]); */
        break;
      case reduce :
        coll.add_reduce(sendbuf_d, 0, recvbuf_d, 0, count * numproc, proclist, ROOT);
        // REDUCE-SCATTER + GATHER
	/* for(int recver = 0; recver < numproc; recver++)
          coll.add_reduce(sendbuf_d, recver * count, recvbuf_d, recver * count, count, proclist, recver);
        coll.add_fence();
        for(int sender = 0; sender < numproc; sender++)
          if(sender != ROOT)
            coll.add_bcast(recvbuf_d, sender * count, recvbuf_d, sender * count, count, sender, ROOT); */
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
      case reducescatter :
        for(int recver = 0; recver < numproc; recver++)
          coll.add_reduce(sendbuf_d, recver * count, recvbuf_d, 0, count, proclist, recver);
        break;
      case allreduce :
        // REDUCE + BROADCAST
        /*coll.add_reduce(sendbuf_d, 0, recvbuf_d, 0, count * numproc, proclist, ROOT);
        coll.add_fence();
        coll.add_bcast(recvbuf_d, 0, recvbuf_d, 0, count * numproc, ROOT, recvids[ROOT]);*/
        // REDUCE-SCATTER + ALL-GATHER
        for(int recver = 0; recver < numproc; recver++)
          coll.add_reduce(sendbuf_d, recver * count, recvbuf_d, recver * count, count, proclist, recver);
        coll.add_fence();
        for(int sender = 0; sender < numproc; sender++)
          coll.add_bcast(recvbuf_d, sender * count, recvbuf_d, sender * count, count, sender, recvids[sender]);
        break;
      default:
        if(myid == ROOT)
          printf("invalid collective option\n");
    }

    // MACHINE DESCRIPTION
    int numlevel = 2;
    int groupsize = numproc / numgroup;
    int hierarchy[5] = {groupsize, 4, 4, 4, 1};
    CommBench::library library[5] = {CommBench::NCCL, CommBench::IPC, CommBench::IPC, CommBench::IPC, CommBench::IPC};
    CommBench::printid = ROOT;

    // INITIALIZE
    double time = MPI_Wtime();
    coll.init(numlevel, hierarchy, library, numstripe, stripeoffset, pipedepth, pipeoffset);
    time = MPI_Wtime() - time;
    if(myid == ROOT)
      printf("preproc time: %e\n", time);

    ExaComm::measure<Type>(count * numproc, warmup, numiter, coll);
    ExaComm::validate(sendbuf_d, recvbuf_d, count, pattern, ROOT, coll);
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
