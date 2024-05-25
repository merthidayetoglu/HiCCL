We first address the most extensive criticisms of the paper.  We address these issues first and then the other points raised.

**Correctness (Rev.3)**

Our method is built on two-sided point-to-point primitives of existing communication libraries - we stress that we are not implementing these lowest-level communication operations ourselves. Thus, issues such as whether receive buffers are filled before the client reads the data do not exist, as we rely on the correctness of these existing implementations.

The correctness of our collective operations depends only on enforcing any needed ordering between multiple point-to-point operations, such as moving data in multiple hops in sequence through the communication hierarchy or performing reductions.  Such operations are dependent on earlier operations, hence the term “dependencies”, which is a standard term to describe such relationships.

We agree happens-before is an important concept, but we do not believe it is appropriate in this context.  Happens-before reasoning is critical in situations with true concurrency; we, however, have a much simpler situation, which is just a dependency graph of operations, again because we assume correct point-to-point operations.

The fence operation is not a barrier. Given two collective operations C1 and C2, the sequence C1; fence; C2 expresses that the components of C2 are pointwise dependent on the components of C1 — i.e., the ith component of C2 must wait until the ith component of C1 has finished (see Figure 4), but not that every component of C2 must wait until every component of C1 has finished. We are certainly open to a better name, and we understand that the explanation can be improved.

**Application Benchmark (Rev.3,4,5)**

We have already integrated our library into PyTorch-DDP and tested it with GPT-2 training on four nodes of Perlmutter. This workload uses a diverse set of collectives with various buffer sizes (12 MB, 25 MB, 50 MB). Switching the DDP backend library is as easy as changing the ‘NCCL’ keyword with ‘HiCCL’ keyword in the training script. Our results show on-par performance (both in terms of per-iteration time and convergence rate).

**Throughput-Oriented Evaluation (Rev.1,2,3)**

In our preliminary studies, we found that MPI collectives with small buffer sizes are sufficiently performant (even more than NCCL/RCCL) in terms of latency. For latency-critical applications, one should simply use MPI functions directly. On the other hand, large message sizes are critical for utilizing accelerators and the network across them. Our study suggests that MPI libraries are not fully optimized for throughput. Therefore we focused on optimizing collective algorithms for throughput. We made this discussion clear in the first three paragraphs of Section 1. We provide latency/bw tradeoff in Fig.9. Paper focuses on >MB regime.

**Rev1**

1) The hierarchical tree structure is automatically built from the user input, specifically through the vector in Listing 2, Line 13. This vector corresponds to the branching factors in each level. Listing 2 is intended for library developers utilizing HiCCL. An informed user can utilize HiCCL directly yet setting the input parameters require expertise.
2) We apologize for not matching the hierarchy in Listing 2 with one of those in Figure 5. Listing 2 is for Aurora, which has six GPUs and each GPU has two dies. Therefore there are 12 endpoints per node in total. When there are two nodes (numproc=24) the factorization in Listing 2 will be the following {2,6,2} (currently not displayed). For clarity, we will replace Figure 5(a) with a display of the parameters in Listing 2.
3) Figure 8 shows the algorithmic throughput of collective functions in isolation.
The geometric mean of HiCCL’s speedup over MPI is calculated based on throughput on four systems and eight collectives that are shown in Figure 8.

**Rev2**

1) The communication policies must be informed by the user through HiCCL’s API (e.g., Listing 2). The intra-node network may consist of multiple levels, e.g., the interconnect across the accelerator dies in a single device and the one across devices differ in Frontier and Aurora. The user must set the intra-node hierarchy according to the dies per device and devices per node as shown in Table V (bold). Moreover, the most performant library implementation in each level can be chosen (among the available) by simply changing the parameters. With a few educated guesses, the optimal parameters will most likely be found. The most effective optimization across nodes is the multi-NIC striping, which can be turned on by simply setting the number of stripes per node.
2) The network architecture with multi-accelerator nodes has converged to hierarchical structures. Specifically for targeting the most recent GPU systems, we designed specialized optimizations for hierarchical systems. Another typical topology is torus (e.g., Tofu), which requires different collective optimizations. The compositional design of HiCCL (with three primitives) can be applied directly to other topologies because of its machine-agnostic nature. However, the factorization of those primitives will be different, and must be specialized for the given network topology.

**Rev3**

**W3,W5,R1,D12,D15.** Refer-to-**A**

**W1.** Refer-to-**C**

**W2.** Fence divides the collective into two steps. Each process wait for completion of the first step before starting the second, hence guaranteeing correctness. Refer-to-**A**.

W4. Yes, the user can tune the parameters Listing2(13–17) for tuning for latency or bandwidth. Refer-to-C.
W6. We exhaustively tested correctness with various 1) compositions, 2) parameters, 3) systems. We describe the verification tests in AD/AE appendix. Refer-to-A.
W7. Refer-to-B.
W8. We will review our terminology usage in the paper and make updates accordingly. We welcome any further feedback from the reviewer in this regard. Please consult our other answers when considering potential alignment with existing terms.
W9. To our knowledge, we have included all the work that directly relates to our contribution in Section VII. We will include happens-before semantics in related work and cite the work pointed out by the reviewer. We are open to adding additional work based on reviewers’ suggestions.
W10. MPI on our test systems are vendor-provided and are optimized/tested extensively for acceptance tests. We worked with facility staff and MPICH developers and set necessary tuning flags to maximize throughput. Refer-to-D8.
W11. Refer-to-B.

R2. Refer-to-A&W9.
R3. a) Refer-to-A b) In practice, the collective performance will be affected by additional communication (if any) and so the theoretical bounds in Table III.

D1. The endpoints of primitives (sendbuf/recvbuf) cannot overlap.
D6. We chose the three primitives for their simplicity and expressivity. Alternative to multicast and reduction would allGatherv and reduceScatterv. However, the alternative has a more complex interface than the original.
D7. We compare ring and tree for broadcast and reduce in Figure 8 to show the case where saturating bandwidth does not mean higher throughput. Similarly, we discuss that an all-reduce with reduction-only primitives is not communication optimal (SIII-Bpar.3,SIV-Cpar.2). On the other hand, a reduce followed by a broadcast (TableIIrow14) is load imbalanced because of reduction on a single GPU. Therefore we chose reduce-scatter followed by all-gather (TabIIrow15), which is communication optimal.
D8. The core algorithms in MPI implementations are originally implemented for CPUs. GPU-aware MPI (OpenMPI/MPICH) typically moves the data to CPU, runs the original algorithm as it is, and then moves the results back to GPU. Therefore they do not take advantage of the direct links across GPUs.
D9. We can also run MVAPICH, but we rely on the available MPI implementations. Refer-to-W10. Regardless, we also show that there is little room for improvement over HiCCL as it already approaches theoretical limits (Figure 8). 
D11. Striping is composed algebraically in SIV-Cpar.2, which can be generalized.
D14. For example, Perlmutter has NVLinks, wher within nodes and SS-11 across nodes. NCCL may be faster within nodes and MPI may be faster across nodes.
D16. Refer-to-D8. The core algorithms in GPU-aware MPI implementations are CPU based. It is hard to compare the ideas in HiCCL with GPU-aware MPI libraries.
D17. Refer-to-W9.
D18. [7] is criticized for costly code synthesis for collective communications, not using SMT in general. We will clarify in the final version.

Rev4
HiCCL relies on non-blocking point-to-point functions of existing GPU-aware communication libraries to implement collective communications. The repository includes a header file that can be modified for integrating additional APIs. For example, a user has integrated the GASNet library as an additional option in one day. Similarly, NVSHMEM, UCX, or another desired library can be integrated as long as it has send/recv or put/get functions that provides a handle for waiting on remote completion.
Yes, please refer to SVI-C2par2. Our implementation exchanges remote memory handles (hipIpcMemHandle_t on Frontier and ze_ipc_mem_handle_t on Aurora) for utilizing interconnects across tiles and devices separately.
Figure 9 shows message sizes on the X axis and throughput on the Y axis for four collective functions on Perlmutter. We observe similar results on other collectives and systems, although we cannot include them in the paper due to space constraints.
We performed the scaling experiment for the sake of stressing the network bandwidth with large messages and finding the limit where the scaling breaks down. We found out the throughput is hampered with message sizes smaller than a MB.
The allreduce algorithm(Fig.2) parallelizes the reduction, where each GPU is responsible to reduce a partial data. With thousands of GPUs, the work per GPU becomes so small that the network and GPU kernel launch latency that would not be significant otherwise becomes significant. It is a future research to find novel compositions to maintain large message sizes at scale. HiCCL’s compositional design will help productivity for developing interesting configurations. For example, one can try employing a single GPU per node for reducing partial data rather than all GPUs for delaying messages getting so small when scaling out.

Rev5
HiCCL’s potential use is to replace one or a few throughput-critical functions manually with the original API(Listing2). Similarly, a drop-in replacement can be achieved with a macro. For legacy applications in Fortran, it is possible to create Fortran bindings of the C++ API.
Refer-to-B.

We will address all minor typographical errors and corrections to figures and text in the final version.
