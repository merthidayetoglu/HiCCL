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

#include "CommBench/commbench.h"

#include <list>
#include <pthread.h>

namespace HiCCL {

  static int printid = 0;
  static const MPI_Comm &comm_mpi = CommBench::comm_mpi;
  static const int &numproc = CommBench::numproc;
  static const int &myid = CommBench::myid;

  static size_t buffsize = 0;
  static size_t recycle = 0;
  static size_t reuse = 0;

  enum pattern {all, others};
  enum collective {dummy, gather, scatter, broadcast, reduce, alltoall, allgather, reducescatter, allreduce};

#include "source/compute.h"
#include "source/coll.h"
#include "source/command.h"
#include "source/reduce.h"
#include "source/broadcast.h"
// #include "source/init.h"
#include "source/comm.h"
#include "source/bench.h"

}

#endif
