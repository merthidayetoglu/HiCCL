# ExaComm

ExaComm is a GPU-Aware communication library. It has a compositional API for separating the collective pattern design from the machine-specific optimizations. It offers hyper primitives to design complicated collective communication patterns. ExaComm then optimizes the primitive pattern for a specified machine, that is described by the user. The implementation is achieved by using the point-to-point functions of the native MPI or NCCL libraries.

For examples, see ```main.cu``` driver code.

For questions and support, please send an email to merth@stanford.edu
