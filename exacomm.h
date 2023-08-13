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
  size_t recycle = 0;
  size_t reuse = 0;

#include "src/compute.h"

  template <typename T>
  class Command {

    CommBench::Comm<T> *comm = nullptr;
    ExaComm::Compute<T> *compute = nullptr;

    public:

    Command(CommBench::Comm<T> *comm) : comm(comm) {}
    Command(ExaComm::Compute<T> *compute) : compute(compute) {}

    void start() {
      if(comm)
        comm->start();
      if(compute)
        compute->start();
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

    // PRIMITIVES
    std::vector<std::vector<BCAST<T>>> bcast_epoch;
    std::vector<std::vector<REDUCE<T>>> reduce_epoch;
    int numepoch = 0;

    // MACHINE
    std::vector<int> numlevel;
    std::vector<std::vector<int>> groupsize;
    std::vector<std::vector<CommBench::library>> library;

    // PIPELINE
    std::vector<std::list<Command<T>>> command_batch;

    public:

    void fence() {
      bcast_epoch.push_back(std::vector<BCAST<T>>());
      reduce_epoch.push_back(std::vector<REDUCE<T>>());
      numepoch++;
    }

    Comm(const MPI_Comm &comm_mpi_temp) : comm_mpi(comm_mpi_temp) {
      // DEFAULT EPOCH
      fence();
    }

    // ADD FUNCTIONS FOR BCAST AND REDUCE PRIMITIVES
    void add_bcast(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
      bcast_epoch[numepoch-1].push_back(BCAST<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvid));
    }
    void add_bcast(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, std::vector<int> &recvids) {
      bcast_epoch[numepoch-1].push_back(BCAST<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvids));
    }
    void add_reduce(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
      reduce_epoch[numepoch-1].push_back(REDUCE<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvid));
    }
    void add_reduce(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, std::vector<int> &sendids, int recvid) {
      reduce_epoch[numepoch-1].push_back(REDUCE<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendids, recvid));
    }

    // INITIALIZE BROADCAST AND REDUCTION TREES
    void init(int numlevel, int groupsize[], CommBench::library lib[], int numbatch, int pipelineoffset) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);

      // ALLOCATE COMMAND BATCH
      for(int batch = 0; batch < numbatch; batch++)
        command_batch.push_back(std::list<Command<T>>());

      if(printid == ROOT) {
        printf("NUMBER OF EPOCHS: %d\n", numepoch);
        for(int epoch = 0; epoch < numepoch; epoch++)
          printf("epoch %d: %zu bcast %zu reduction\n", epoch, bcast_epoch[epoch].size(), reduce_epoch[epoch].size());
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

      // FOR EACH EPOCH
      for(int epoch = 0; epoch < numepoch; epoch++) {
        // INIT REDUCTION
        std::vector<REDUCE<T>> &reducelist = reduce_epoch[epoch];
        if(reducelist.size())
        {
          // PARTITION INTO BATCHES
          std::vector<std::vector<REDUCE<T>>> reduce_batch(numbatch);
          ExaComm::batch(reducelist, numbatch, reduce_batch);
          // FOR EACH BATCH
          for(int batch = 0; batch < numbatch; batch++)
            if(groupsize[0] < numlevel) {
              // HIERARCHICAL REDUCTION RING
            }
            else {
              // HIERARCHICAL REDUCTION TREE
              std::vector<T*> recvbuff; // for memory recycling
              ExaComm::reduce_tree(comm_mpi, numlevel, groupsize, lib, reduce_batch[batch], numlevel - 1, command_batch[batch], recvbuff, 0);
            }
        }
        // INIT BROADCAST
        std::vector<BCAST<T>> &bcastlist = bcast_epoch[epoch];
        if(bcastlist.size())
        {
          // PARTITION INTO BATCHES
          std::vector<std::vector<BCAST<T>>> bcast_batch(numbatch);
          ExaComm::batch(bcastlist, numbatch, bcast_batch);
          // FOR EACH BATCH
          for(int batch = 0; batch < numbatch; batch++)
            if(groupsize[0] < numlevel) {
              // HIERARCHICAL BROADCAST RING
            }
            else {
              // HIERARCHICAL BROADCAST TREE
              ExaComm::bcast_tree(comm_mpi, numlevel, groupsize, lib, bcast_batch[batch], 1, command_batch[batch]);
            }
        }
      }
      // INITIALIZE BATCH PIPELINE WITH DUMMY COMMANDS
      {
        for(int batch = 0; batch < numbatch; batch++)
          for(int c = 0; c < batch * pipelineoffset; c++)
            command_batch[batch].push_front(ExaComm::Command<T>(new CommBench::Comm<T>(comm_mpi, CommBench::MPI)));
      }
      // REPORT MEMORY
      {
        std::vector<size_t> buffsize_all(numproc);
        std::vector<size_t> recycle_all(numproc);
        std::vector<size_t> reuse_all(numproc);
        MPI_Allgather(&buffsize, sizeof(size_t), MPI_BYTE, buffsize_all.data(), sizeof(size_t), MPI_BYTE, comm_mpi);
        MPI_Allgather(&recycle, sizeof(size_t), MPI_BYTE, recycle_all.data(), sizeof(size_t), MPI_BYTE, comm_mpi);
        MPI_Allgather(&reuse, sizeof(size_t), MPI_BYTE, reuse_all.data(), sizeof(size_t), MPI_BYTE, comm_mpi);
        if(printid == ROOT) {
          for(int p = 0; p < numproc; p++)
            printf("ExaComm Memory [%d]: %zu bytes (%.2f GB) - %.2f GB reused - %.2f GB recycled\n", p, buffsize_all[p] * sizeof(T), buffsize_all[p] * sizeof(T) / 1.e9, reuse_all[p] * sizeof(T) / 1.e9, recycle_all[p] * sizeof(T) / 1.e9);
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
          if(commandptr[i] != command_batch[i].end()) {
            commandptr[i]->start();
          }
        for(int i = 0; i < command_batch.size(); i++)
          if(commandptr[i] != command_batch[i].end()) {
            commandptr[i]->wait();
          }
        for(int i = 0; i < command_batch.size(); i++)
          if(commandptr[i] != command_batch[i].end()) {
            finished = false;
            commandptr[i]++;
          }
      }
    }

    void measure(int warmup, int numiter) {
      if(printid == ROOT) {
        printf("command_batch size %zu\n", command_batch.size());
        printf("commandlist size %zu\n", command_batch[0].size());
      }
      auto it = command_batch[0].begin();
      for(int command = 0; command < command_batch[0].size(); command++) {
        it->measure(warmup, numiter);
        it++;
      }
    }

    void report() {
      if(printid == ROOT) {
        printf("command_batch size %zu\n", command_batch.size());
        printf("commandlist size %zu\n", command_batch[0].size());
      }
      int counter = 0;
      for(auto it = command_batch[0].begin(); it != command_batch[0].end(); it++) {
        if(printid == ROOT) {
          printf("counter: %d command::", counter);
        }
        it->report();
        counter++;
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
