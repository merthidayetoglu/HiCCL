#define PORT_HIP
#include "hiccl.h"

#define Type size_t

int main() {

  int myid = CommBench::myid;
  int numproc = CommBench::numproc;
  int root = 0;

  size_t count = 128e6; // 1 GB per GPU
  size_t *weights;
  CommBench::allocate(weights, count);

  HiCCL::Comm<Type> allreduce;
  {
    // END POINTS
    Type *sendbuf;
    Type *recvbuf;
    CommBench::allocate(sendbuf, count);
    CommBench::allocate(recvbuf, count) ;

    // COMPOSITION
/*
    // DIRECT REDUCE
    for(int recver = 0; recver < numproc; recver++)
      allreduce.add_reduce(sendbuf, 0, recvbuf, 0, count, HiCCL::all, recver);
    // REDUCE + BROADCAST
    allreduce.add_reduce(sendbuf, 0, recvbuf, 0, count, HiCCL::all, root); // all -> root
    allreduce.add_fence();
    allreduce.add_bcast(recvbuf, 0, recvbuf, 0, count, root, HiCCL::others); // root -> all - root
    // REDUCE + BROADCAST OMITTING THE ROOT REDUCTION
    int nodesize = 8;
    for(int i = 0; i < nodesize; i++)
      allreduce.add_reduce(sendbuf, i * (count / nodesize), recvbuf, i * (count / nodesize), count / nodesize, HiCCL::all, i);
    allreduce.add_fence();
    for(int i = 0; i < nodesize; i++)
      allreduce.add_bcast(recvbuf, i * (count / nodesize), recvbuf, i * (count / nodesize), count / nodesize, i, HiCCL::others);  
*/
    // REDUCE-SCATTER + ALL-GATHER
    for(int recver = 0; recver < numproc; recver++)
      allreduce.add_reduce(sendbuf, recver * count / numproc, recvbuf, recver * count / numproc, count / numproc, HiCCL::all, recver);
    allreduce.add_fence();
    for(int sender = 0; sender < numproc; sender++)
      allreduce.add_bcast(recvbuf, sender * count / numproc, recvbuf, sender * count / numproc, count / numproc, sender, HiCCL::others);
    
/*
*/
    // SET PARAMETERS
    allreduce.set_hierarchy(std::vector<int> {2, 4, 2}, std::vector<CommBench::library> {CommBench::MPI, CommBench::IPC, CommBench::IPC});
    // allreduce.set_numstripe(8);
    // allreduce.set_ringnodes(16);
    allreduce.set_pipedepth(4);
    allreduce.set_endpoints(sendbuf, count, recvbuf, count);
    // CommBench::printid = -1;
    allreduce.init();
    // CommBench::printid = 0;
    CommBench::report_memory();

    HiCCL::validate(sendbuf, recvbuf, count / numproc,  HiCCL::allreduce, root, allreduce);
  }

  allreduce.run(weights, weights);

  HiCCL::measure(5, 10, count, allreduce);

  CommBench::free(weights);

  return 0;
}
