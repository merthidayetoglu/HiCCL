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

#include "../CommBench/comm.h"

#include <vector>
#include <list>
#include <iterator>
#include <numeric>

namespace ExaComm {

  int printid;
  FILE *pFile;
  size_t buffsize = 0;

#include "src/comp.h"

  template <typename T>
  class Command {

    CommBench::Comm<T> *comm = nullptr;
    ExaComm::Compute<T> *compute = nullptr;

    public:

    Command(CommBench::Comm<T> *comm) : comm(comm) {}
    Command(ExaComm::Compute<T> *compute) : compute(compute) {}
    /*Command(Command &t) {
      comm = t.comm;
      compute = t.compute;
    }*/

    void start() {
      if(comm) comm->start();
      if(compute) compute->start();
    }
    void wait() {
      if(comm)
        comm->wait();
      if(compute)
        compute->wait();
    }
    void run() { start(); wait(); }
    void report() {
      if(comm) {
        if(printid == ROOT)
          printf("COMMAND TYPE: COMMUNICATION\n");
        comm->report();
      }
      if(compute) {
        if(printid == ROOT)
          printf("COMMAND TYPE: COMPUTATION\n");
        compute->report();
      }
    }
    void measure(int warmup, int numiter) {
      if(comm) {
        if(printid == ROOT)
          printf("COMMAND TYPE: COMMUNICATION\n");
	comm->measure(warmup, numiter);
      }
      if(compute) {
        if(printid == ROOT)
          printf("COMMAND TYPE: COMPUTATION\n");
	compute->measure(warmup, numiter);
      }
    }
  };

#define FUSING
#include "src/bcast.h"
#include "src/reduce.h"

  template <typename T>
  class Comm {

    const MPI_Comm comm_mpi;

    std::vector<BCAST<T>> bcastlist;
    std::vector<REDUCE<T>> reducelist;

    // PIPELINING
    std::vector<std::list<Command<T>>> command_batch;

    public:

    Comm(const MPI_Comm &comm_mpi_temp) : comm_mpi(comm_mpi_temp) {}

    // ADD FUNCTIONS FOR BCAST AND REDUCE PRIMITIVES
    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
      bcastlist.push_back(BCAST<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvid));
    }
    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, std::vector<int> &recvids) {
      bcastlist.push_back(BCAST<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvids));
    }
    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, std::vector<int> &sendids, int recvid) {
      reducelist.push_back(REDUCE<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendids, recvid));
    }

    // INITIALIZE BROADCAST AND REDUCTION TREES
    void init(int numlevel, int groupsize[], CommBench::library lib[], int numbatch) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);

      // ALLOCATE COMMAND BATCH
      for(int batch = 0; batch < numbatch; batch++)
        command_batch.push_back(std::list<Command<T>>());

      if(printid == ROOT) {
        printf("Initialize ExaComm with %d levels\n", numlevel);
        for(int level = 0; level < numlevel; level++) {
          printf("level %d groupsize %d library: ", level, groupsize[level]);
          switch(lib[level]) {
            case(CommBench::IPC) :
              printf("IPC");
              break;
            case(CommBench::MPI) :
              printf("MPI");
              break;
            case(CommBench::NCCL) :
              printf("NCCL");
              break;
          }
          if(level == 0)
            if(groupsize[0] != numproc)
              printf(" *");
          printf("\n");
        }
        printf("\n");
      }
      // INIT BROADCAST
      if(bcastlist.size())
      {
        // PARTITION BROADCAST INTO BATCHES
        std::vector<std::vector<BCAST<T>>> bcast_batch(numbatch);
        {
          for(auto &bcast : bcastlist) {
            int batchsize = bcast.count / numbatch;
            for(int batch = 0; batch < numbatch; batch++)
              bcast_batch[batch].push_back(BCAST<T>(bcast.sendbuf, bcast.sendoffset + batch * batchsize, bcast.recvbuf, bcast.recvoffset + batch * batchsize, batchsize, bcast.sendid, bcast.recvids));
          }
        }
	if(groupsize[0] < numproc) {
	  // SCATTER
	  for(int batch = 0; batch < numbatch; batch++)
	    ExaComm::scatter(comm_mpi, groupsize[0], lib[0], lib[numlevel-1], bcast_batch[batch], command_batch[batch]);
          // STRIPE BROADCAST
          //for(int batch = 0; batch < numbatch; batch++)
	  //  ExaComm::stripe(comm_mpi, groupsize[0], lib[numlevel-1], bcast_batch[batch], command_batch[batch]);
        }

        // ALLGATHER
        std::vector<int> groupsize_temp(groupsize, groupsize + numlevel);
        groupsize_temp[0] = numproc;
        for(int batch = 0; batch < numbatch; batch++)
          ExaComm::bcast_tree(comm_mpi, numlevel, groupsize_temp.data(), lib, bcast_batch[batch], 1, command_batch[batch]);
      }
      // ADD INITIAL DUMMY COMMUNICATORS INTO THE PIPELINE
      if(bcastlist.size() | reducelist.size()) {
        for(int batch = 0; batch < numbatch; batch++)
          for(int c = 0; c < batch; c++)
            command_batch[batch].push_front(ExaComm::Command<T>(new CommBench::Comm<T>(comm_mpi, CommBench::MPI)));
      }

      // REPORT
      {
        std::vector<size_t> buffsize_all(numproc);
        MPI_Allgather(&buffsize, sizeof(size_t), MPI_BYTE, buffsize_all.data(), sizeof(size_t), MPI_BYTE, comm_mpi);
        if(printid == ROOT) {
          for(int p = 0; p < numproc; p++)
            printf("ExaComm Memory [%d]: %zu bytes\n", p, buffsize_all[p] * sizeof(size_t));
          printf("command_batch size %zu: ", command_batch.size());
          for(int i = 0; i < command_batch.size(); i++)
            printf("%zu ", command_batch[i].size());
          printf("\n\n");
        }
      }

    };

    void run() {
      using Iter = typename std::list<ExaComm::Command<T>>::iterator;
      std::vector<Iter> commandptr(command_batch.size());
      for(int i = 0; i < command_batch.size(); i++)
        commandptr[i] = command_batch[i].begin();

      bool finished = false;
      while(!finished) {
        finished = true;
        for(int i = 0; i < command_batch.size(); i++)
          if(commandptr[i] != command_batch[i].end())
            commandptr[i]->start();
        for(int i = 0; i < command_batch.size(); i++)
          if(commandptr[i] != command_batch[i].end())
            commandptr[i]->wait();
        for(int i = 0; i < command_batch.size(); i++)
          if(commandptr[i] != command_batch[i].end()) {
            commandptr[i]++;
            finished = false;
          }
      }
    }

    void measure(int warmup, int numiter) {
      for(auto &command : command_batch[0])
        command.measure(warmup, numiter);
    }

    void report() {
      int counter = 0;
      for(auto it = command_batch[0].begin(); it != command_batch[0].end(); it++) {
        if(printid == ROOT) {
          printf("counter: %d command::", counter);
        }
        it->report();
        counter++;
      }
      if(printid == ROOT) {
        printf("command_batch size %zu\n", command_batch.size());
        printf("commandlist size %zu\n", command_batch[0].size());
      }
    }
  };

#include "src/bench.h"

  /*template <typename T>
  void run_concurrent(std::vector<std::list<CommBench::Comm<T>*>> &commlist) {

    using Iter = typename std::list<CommBench::Comm<T>*>::iterator;
    std::vector<Iter> commptr(commlist.size());
    for(int i = 0; i < commlist.size(); i++)
      commptr[i] = commlist[i].begin();

    for(int i = 0; i < commlist.size(); i++)
      if(commptr[i] != commlist[i].end()) {
        // fprintf(pFile, "start i %d init\n", i);
        (*commptr[i])->start();
      }

    bool finished = false;
    while(!finished) {
      finished = true;
      for(int i = 0; i < commlist.size(); i++) {
        if(commptr[i] != commlist[i].end()) {
          if(!(*commptr[i])->test()) {
            // fprintf(pFile, "test %d\n", i);
            finished = false;
          }
          else {
            // fprintf(pFile, "wait %d\n", i);
            (*commptr[i])->wait();
            commptr[i]++;
            if(commptr[i] != commlist[i].end()) {
              // fprintf(pFile, "start next %d\n", i);
              (*commptr[i])->start();
              finished = false;
            }
            else {
              ; //fprintf(pFile, "i %d is finished\n", i);
            }
          }
        }
      }
    }
  }*/

/*  template <typename T>
  void run_commandlist(std::list<Command<T>> &commandlist) {
    for(auto comm : commandlist) {
      switch(comm.com) {
        case(command::start) :
          comm.comm->start();
          break;
        case(command::wait) :
          comm.comm->wait();
          break;
      }
    }
  }

  template <typename T>
  void run_commlist(std::list<CommBench::Comm<T>*> &commlist) {
    for(auto comm : commlist) {
      comm->run();
    }
  }*/
}
