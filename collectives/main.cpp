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

// #define PORT_SYCL
#define PORT_HIP
// #define PORT_CUDA
#include "../hiccl.h"

#define ROOT 0

// USER DEFINED TYPE
#define Type size_t
/*struct Type
{
  // int tag;
  int data[1];
  // complex<double> x, y, z;
};*/

int main(int argc, char *argv[])
{
  // INITIALIZE
  CommBench::init();
  int myid = CommBench::myid;
  int numproc = CommBench::numproc;
  // MPI_Init(&argc, &argv);
  // MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  // MPI_Comm_size(MPI_COMM_WORLD, &numproc);
  // char machine_name[MPI_MAX_PROCESSOR_NAME];
  // int name_len = 0;
  // MPI_Get_processor_name(machine_name, &name_len);
  // printf("myid %d %s\n",myid, machine_name);

  // INPUT PARAMETERS
  int pattern = atoi(argv[1]);
  size_t count = atol(argv[2]);
  int numstripe = atoi(argv[3]);
  int ringnodes = atoi(argv[4]);
  int pipedepth = atoi(argv[5]);
  int warmup = atoi(argv[6]);
  int numiter = atoi(argv[7]);


  // PRINT NUMBER OF PROCESSES AND THREADS
  if(myid == CommBench::printid)
  {
    printf("\n");
    printf("Number of processes: %d\n", numproc);
    printf("Number of warmup %d\n", warmup);
    printf("Number of iterations %d\n", numiter);
    printf("\n");

    printf("Pattern: ");
    switch(pattern) {
      case HiCCL::gather        : printf("Gather\n");         break;
      case HiCCL::scatter       : printf("Scatter\n");        break;
      case HiCCL::broadcast     : printf("Broadcast\n");      break;
      case HiCCL::reduce        : printf("Reduce\n");         break;
      case HiCCL::alltoall      : printf("All-to-All\n");     break;
      case HiCCL::allgather     : printf("All-Gather\n");     break;
      case HiCCL::reducescatter : printf("Reduce-Scatter\n"); break;
      case HiCCL::allreduce     : printf("All-Reduce\n");     break;
    }
    printf("\n");
    printf("Bytes per Type %lu\n", sizeof(Type));
    printf("count %ld: ", count);
    CommBench::print_data(count * sizeof(Type));
    printf("\n");
    {
      size_t data = 2 * count * numproc * sizeof(Type);
      printf("sendbuf + recvbuf count: %zu + %zu = ", count * numproc, count * numproc);
      CommBench::print_data(data);
      printf("\n");
    }
    printf("Number of stripes: %d\n", numstripe);
    printf("Number of ring nodes: %d\n", ringnodes);
    printf("Pipeline depth: %d\n", pipedepth);
  }

  // ALLOCATE
  Type *sendbuf_d;
  Type *recvbuf_d;
  CommBench::allocate(sendbuf_d, count * numproc);
  CommBench::allocate(recvbuf_d, count * numproc);

  // COLLECTIVE COMMUNICATION
  {
    HiCCL::Comm<Type> coll;

    HiCCL::printid = -1;    
    // PATTERN DESRIPTION
    switch (pattern) {
	    case HiCCL::gather :
        for(int sender = 0; sender < numproc; sender++)
          coll.add_bcast(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, ROOT);
        break;
      case HiCCL::scatter :
        for(int recver = 0; recver < numproc; recver++)
          coll.add_reduce(sendbuf_d, recver * count, recvbuf_d, 0, count, ROOT, recver);
        break;
      case HiCCL::broadcast :
        coll.add_bcast(sendbuf_d, 0, recvbuf_d, 0, count * numproc, ROOT, HiCCL::all);
        // SCATTER + ALL-GATHER
        /* for(int recver = 0; recver < numproc; recver++)
          coll.add_reduce(sendbuf_d, recver * count, recvbuf_d, recver * count, count, ROOT, recver);
        coll.add_fence();
        for(int sender = 0; sender < numproc; sender++)
          coll.add_bcast(recvbuf_d, sender * count, recvbuf_d, sender * count, count, sender, HiCCL::others); */
        break;
      case HiCCL::reduce :
        coll.add_reduce(sendbuf_d, 0, recvbuf_d, 0, count * numproc, HiCCL::all, ROOT);
        // REDUCE-SCATTER + GATHER
	/* for(int recver = 0; recver < numproc; recver++)
          coll.add_reduce(sendbuf_d, recver * count, recvbuf_d, recver * count, count, HiCCL::all, recver);
        coll.add_fence();
        for(int sender = 0; sender < numproc; sender++)
          if(sender != ROOT)
            coll.add_bcast(recvbuf_d, sender * count, recvbuf_d, sender * count, count, sender, ROOT); */
        break;
      case HiCCL::alltoall :
        for(int sender = 0; sender < numproc; sender++)
          for(int recver = 0; recver < numproc; recver++)
            coll.add_bcast(sendbuf_d, recver * count, recvbuf_d, sender * count, count, sender, recver);
        break;
      case HiCCL::allgather :
        for(int sender = 0; sender < numproc; sender++)
          coll.add_bcast(sendbuf_d, 0, recvbuf_d, sender * count, count, sender, HiCCL::all);
        break;
      case HiCCL::reducescatter :
        for(int recver = 0; recver < numproc; recver++)
          coll.add_reduce(sendbuf_d, recver * count, recvbuf_d, 0, count, HiCCL::all, recver);
        break;
      case HiCCL::allreduce :
        // REDUCE + BROADCAST
        /*coll.add_reduce(sendbuf_d, 0, recvbuf_d, 0, count * numproc, HiCCL::all, ROOT);
        coll.add_fence();
        coll.add_bcast(recvbuf_d, 0, recvbuf_d, 0, count * numproc, ROOT, HiCCL::others);*/
        // REDUCE-SCATTER + ALL-GATHER
        for(int recver = 0; recver < numproc; recver++)
          coll.add_reduce(sendbuf_d, recver * count, recvbuf_d, recver * count, count, HiCCL::all, recver);
        coll.add_fence();
        for(int sender = 0; sender < numproc; sender++)
          coll.add_bcast(recvbuf_d, sender * count, recvbuf_d, sender * count, count, sender, HiCCL::others);
        break;
      default:
        if(myid == CommBench::printid)
          printf("invalid collective option\n");
    }
    HiCCL::printid = 0;    

    // INITIALIZE
     coll.set_hierarchy(std::vector<int> {4, 4, 2},
                        std::vector<CommBench::library> {CommBench::MPI, CommBench::IPC, CommBench::IPC});
    //coll.set_hierarchy(std::vector<int> {2, 2, 4, 2},
     //                  std::vector<CommBench::library> {CommBench::MPI, CommBench::MPI, CommBench::IPC, CommBench::IPC});
    // coll.set_hierarchy(std::vector<int> {32, 8},
    //                    std::vector<CommBench::library> {CommBench::MPI, CommBench::IPC});
    coll.set_numstripe(numstripe);
    coll.set_ringnodes(ringnodes);
    coll.set_pipedepth(pipedepth);

    CommBench::printid = -1;
    coll.init();
    CommBench::printid = 0;

    coll.measure(warmup, numiter, count * numproc / pipedepth);

    CommBench::report_memory();
    HiCCL::measure<Type>(warmup, numiter, count * numproc, coll);
    HiCCL::validate(sendbuf_d, recvbuf_d, count, pattern, ROOT, coll);
  }
  if(myid == CommBench::printid) {
    printf("approx. message length: ");
    CommBench::print_data((((double)count / numstripe) / pipedepth) * sizeof(Type));
    printf("\n");
  }

// DEALLOCATE
  CommBench::free(sendbuf_d);
  CommBench::free(recvbuf_d);

  return 0;
} // main()
