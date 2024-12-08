**Reviever 1**

**Reviever 2**

**Reviever 3**

**Reviever 4**

**Integration of a new API.**
HiCCL’s is designed for easy integration of new library APIs for mixed-library implementation. The collectives are ultimately implemented with point-to-point functions, and HiCCL takes advantage of non-blocking point-to-point API of a new communication library via a simplified interface. We used that interface to integrate the existing libraries–NCCL, MPI, IPC (CUDA/HIP/OneAPI). In fact, we have recently integrated GASNet for non-MPI applications in one day of engineering effort. In the end, the user can choose whichever library they want in a particular hierarchy level as in Line 14 of Listing 2.

**Intra-node communication hierarchies.**
The key contribution of this work is the abstraction of the communication hierarchies. HiCCL takes inter-tile and inter-GPU interconnects into account as explicitly stated in the 4th sentence of 2nd paragraph of Section VI. C. 2). The overall hierarchy is set using a vector as in Line 13 of Listing 2. In the Aurora example, the last two elements {6, 2} represent six devices (connected with XeLinks) with two tiles (connected with MDFI). In evaluation, we will include this detail and refer to Line 13 of Listing 2 for convenience of the reader.

**Message size vs. bandwidth.**
We show the throughput for various message sizes in Figure 9 for a few representative cases. Other systems / collectives look similar, but omitted because of space constraints.

**Strong scaling.**
We did not choose the message sizes based on any specific application. We chose them for the sake of stressing the network bandwidth with large messages and to investigate if we can reach the theoretical limits.

**Latency.**
In allgather (Fig.2), each GPU is responsible to reduce partial data. With thousands of GPUs, the work per GPU becomes so small (<MB) that the network and GPU kernel launch latency becomes significant. It is future research to find novel compositions to maintain large message sizes at scale.