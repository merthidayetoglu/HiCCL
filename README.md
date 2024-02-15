# HiCCL

HiCCL is a hierarchical communication library. It has a compositional API for separating the collective pattern design from the machine-specific optimizations. ExaComm optimizes the given high-level pattern for a specified machine, that is described by the user. The implementation is achieved by using the point-to-point functions of the native communication libraries, such as MPI and NCCL.

For questions and support, please send an email to merth@stanford.edu
