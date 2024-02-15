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

#ifndef HICCL_H
#define HICCL_H

// GPU PORTS
// For NVIDIA: #define PORT_CUDA
// For AMD: #define PORT_HIP
// For SYCL: #define PORT_SYCL

#include "CommBench/comm.h"

#include <list>

namespace HiCCL {

  static const MPI_Comm &comm_mpi = CommBench::comm_mpi;
  static const int &printid = CommBench::printid;
  static const int &numproc = CommBench::numproc;
  static const int &myid = CommBench::myid;

  static size_t buffsize = 0;
  static size_t recycle = 0;
  static size_t reuse = 0;

#include "source/compute.h"
#include "source/coll.h"
#include "source/command.h"
#include "source/reduce.h"
#include "source/broadcast.h"
#include "source/comm.h"
#include "source/bench.h"

  template <typename T>
  Comm<T> init_allreduce(size_t count, std::vector<int> hierarchy, std::vector<CommBench::library> library, int pipeline) {
    T *commbuf;
    CommBench::allocate(commbuf, count);
    T *tempbuf;
    if(myid == 0)
      CommBench::allocate(tempbuf, count);
    std::vector<int> proclist;
    for(int i = 0; i < numproc; i++)
      proclist.push_back(i);
    Comm<T> allreduce;
    allreduce.add_reduce(commbuf, 0, tempbuf, 0, count, proclist, 0);
    allreduce.add_fence();
    allreduce.add_bcast(tempbuf, 0, commbuf, 0, count, 0, proclist);

    int numlevel = hierarchy.size();
    std::vector<int> groupsize(numlevel);
    groupsize[numlevel - 1] = hierarchy.back();
    for(int level = numlevel - 2; level > -1; level--)
      groupsize[level] = groupsize[level - 1] * hierarchy[level];
    allreduce.init(numlevel, groupsize.data(), library.data(), 1, 1, pipeline, 1);

    CommBench::report_memory();

    allreduce.sendbuf = commbuf;
    allreduce.sendcount = count;
    allreduce.recvbuf = commbuf;
    allreduce.recvcount = count;

    return allreduce;
  }
}

#endif
