We first address the main criticisms of the paper and then the other points raised.

**Correctness (Rev.3)**

Our method is built on two-sided point-to-point primitives of existing communication libraries - we do not implement these lowest-level communication operations ourselves. Thus, issues such as whether receive buffers are filled before the client reads the data don’t exist, as we rely on the correctness of these existing implementations.

The correctness of our collective operations depends only on enforcing any needed ordering between multiple point-to-point operations, such as when moving data in multiple hops in sequence through the communication hierarchy or performing reductions.  Such operations are dependent on earlier operations, hence “dependencies”, which is a standard term to describe such relationships.

Happens-before reasoning is critical in situations with true concurrency or imprecise dependencies.  Our situation is a precise dependence graph of operations–every edge is a guaranteed data dependence with no approximation, and no other dependencies are possible. Thus, simply enforcing data dependences is both necessary and sufficient for correctness.  

The fence operation is not a barrier. Given two collective operations C1 and C2, the sequence “C1; fence; C2” expresses that the components of C2 are pointwise dependent on the components of C1 —the ith component of C2 must wait until the ith component of C1 has finished (see Figure 4), but not that every component of C2 must wait until every component of C1 has finished. We are open to a better name, and we accept that the explanation can be improved.

**Application Benchmark (Rev.3,4,5)**

We integrated our library into PyTorch-DDP and tested it with GPT-2 training on four nodes of Perlmutter. This workload uses various buffer sizes (12 MB, 25 MB, 50 MB). Switching the DDP backend requires changing the ‘NCCL’ keyword with ‘HiCCL’ keyword in the training script. Our results show on-par performance with NCCL’s (both in terms of per-iteration time and convergence rate).

**Throughput-Oriented Evaluation (Rev.1,2,3)**

Our preliminary studies have shown that one should simply use MPI for latency-critical applications. However, large message sizes are critical for utilizing accelerators, which is our focus; see the first three paragraphs of Section 1. 

**Parameter Selection (Rev.1,2,4)**

HiCCL’s API (Listing 2) is intended for library developers. Communication policies are informed by the developer. For example, the user must set the intra-node hierarchy according to the dies per device and devices per node as shown in Table V (bold). The hierarchical tree structure is automatically built from the user input as explained in Sec.IV.

**Rev1**

1) 2) We apologize, the hierarchy in Listing 2 is not displayed in Fig.5. Listing 2 is for Aurora with six GPUs and each GPU has two dies. Therefore two nodes (numproc=24) are factored as {2,6,2}. For clarity, we will replace Fig.5(a) with a display of {2,6,2}.
3) Fig.8 compares the algorithmic throughput of collective functions in isolation.
4) The geometric mean of HiCCL’s speedup over MPI is based on throughput across four systems and eight collectives shown in Fig.8.

**Rev2**

All GPU systems that we know today have hierarchical networks.

Background explains achievable bandwidth of 75% due to load imbalance across NICs, which manifests in Aurora as shown in Fig.8(d) with “not achievable” frames.

**Rev3**

**W1**,**W4**. Refer-to-**C**
**W2**,**W3**,**W5**,**W6**,**R1**,**D12**,**D15**. Refer-to-**A**
**W7**,**W11**. Refer-to-**B**.
**W9**,**D17**. To our knowledge, we have included the work that directly relates to our contribution. We will include happens-before semantics in related work and cite the work pointed out by the reviewer. We are open to adding additional work and updating terminology based on reviewers’ suggestions.

**R2.** Refer-to-**A**&**W9**.

**R3**. **a)** Refer-to-**A** **b)** Collective performance will be impacted by additional communication, and so the theoretical bounds in Table III.

**D7**. Fig.8 compares ring and tree for broadcast and reduce, showing that saturating bandwidth does not mean higher throughput. Similarly, we discuss that an all-reduce with reduction-only primitives is sub-optimal (Sec.III-Bpar.3,Cpar.2). Therefore we chose reduce-scatter followed by all-gather (TabIe2row15), which is communication optimal.
D8,D16,W10. The core algorithms in GPU-aware MPI implementations are originally developed for CPUs. Current implementations (OpenMPI/MPICH) move the data to CPU, run an original algorithm as is, and then move the results back to GPU. Therefore they do not take advantage of the direct links across GPUs, and it is hard to compare the ideas in HiCCL. We confirmed our discussion with MPICH developers.
D18. [7] is criticized for costly code synthesis for collective communications. We will clarify in the final version.

We will address all detailed comments in the paper, but due to space constraints cannot answer exhaustively.

Rev4

New API integration: The HiCCL code includes a header file that can be modified for integrating new APIs’s point-to-point functions. Once integrated, the new API can be chosen to be used at any level. For example, a user has integrated the GASNet library as an additional option in one day.

Intra-node/intra-device links: Please refer to SecVI-C2par2. Our implementation chooses appropriate copy engines for efficiently utilizing interconnect across dies and devices.

Buffer size vs. bandwidth: Fig.9 characterizes this for four collective functions on Perlmutter. We observe similar results on other collectives and systems; we can include such results given more space.

Scaling: We performed the scaling experiment for the sake of stressing the network bandwidth with large messages and finding the limit where the scaling breaks down. In allgather (Fig.2), each GPU is responsible to reduce partial data. With thousands of GPUs, the work per GPU becomes so small (<MB) that the network and GPU kernel launch latency becomes significant. It is future research to find novel compositions to maintain large message sizes at scale.

Rev5

HiCCL integration: HiCCL’s typical use is replacing throughput-critical functions manually with the original API(Listing2). Alternatively, a drop-in replacement can be achieved with compiler macro. For legacy applications in Fortran, it is possible to create Fortran bindings of the C++ API. We developed Python bindings as a proof of concept.

We will address all minor typographical errors and corrections to figures and text in the final version.
