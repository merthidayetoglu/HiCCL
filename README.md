# HiCCL

HiCCL is a hierarchical communication library. It has a compositional API for separating the collective pattern design from the machine-specific optimizations. HiCCL optimizes the given high-level pattern for a specified machine that is described by the user. The implementation is achieved by using the point-to-point functions of the native communication libraries, such as IPC (put, get), MPI, NCCL, RCCL, and OneCCL.

For questions and support, please send an email to merth@stanford.edu
