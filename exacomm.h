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

#ifndef EXACOMM_H
#define EXACOMM_H

#include "CommBench/comm.h"

#include <vector>
#include <list>

namespace ExaComm {

  const MPI_Comm &comm_mpi = CommBench::comm_mpi;
  int printid = -1;

  size_t buffsize = 0;
  size_t recycle = 0;
  size_t reuse = 0;

#include "source/compute.h"
#include "source/coll.h"
#include "source/command.h"
#include "source/reduce.h"
#include "source/broadcast.h"
#include "source/comm.h"
#include "source/bench.h"

}

#endif
