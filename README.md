# HiCCL

HiCCL is a GPU-Aware communication library. It has a compositional API for separating the collective pattern design from the machine-specific optimizations. It offers the Expand and Fold primitives to design complicated collective communication patterns. HiCCL then optimizes the primitive pattern for a specified machine, that is described by the user.

As an example, composition and optimization of all-reduce function is given below.

```cpp
#define PORT_HIP
#include "exacomm.h"

...

  ExaComm::Comm<float> allreduce;

  int numgpu_total;
  MPI_Comm_size(&numgpu_total, ExaComm::comm_mpi);
  std::vector<int> sendids;
  std::vector<std::vector<int>> recvids(numgpu_total);
  for(int sender = 0; sender < numgpu_total; sender++) {
    sendids.push_back(sender);
    for(int recver = 0; recver < numgpu_total; recver++)
      if(recver != sender)
        recvids[sender].push_back(recver);
  }
  
  // Composition as reduce-scatter + all-gather
  for(int recver = 0; recver < numgpu_total; recver++)
    coll.add_reduce(sendbuf_d, recver * count, recvbuf_d, recver * count, count, proclist, recver);
  coll.add_fence();
  for(int sender = 0; sender < numproc; sender++)
    coll.add_bcast(recvbuf_d, sender * count, recvbuf_d, sender * count, count, sender, recvids[sender]);

...


```

For questions and support, please send an email to merth@stanford.edu
