# HiCCL

HiCCL is a GPU-Aware communication library. It has a compositional API for separating the collective pattern design from the machine-specific optimizations. It offers the Expand and Fold primitives to design complicated collective communication patterns. HiCCL then optimizes the primitive pattern for a specified machine, that is described by the user.

As an example, composition and optimization of all-reduce function is given below.

```c++
#define PORT_HIP
#include "exacomm.h"

ExaComm::Comm<float> allreduce;

int numgpu_total;
MPI_Comm_size(&numgpu_total, ExaComm::comm_mpi);
int numgpu_node = 8;

// Composition as reduce-scatter + all-gather


```

For questions and support, please send an email to merth@stanford.edu
