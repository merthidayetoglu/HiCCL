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
  class Coll {

     public:

     CommBench::library lib;

     // Communication
     int numcomm = 0;
     std::vector<T*> sendbuf;
     std::vector<size_t> sendoffset;
     std::vector<T*> recvbuf;
     std::vector<size_t> recvoffset;
     std::vector<size_t> count;
     std::vector<int> sendid;
     std::vector<int> recvid;

     // Computation
     int numcompute = 0;
     std::vector<std::vector<T*>> inputbuf;
     std::vector<T*> outputbuf;
     std::vector<size_t> numreduce;
     std::vector<int> compid;

     Coll(CommBench::library lib) : lib(lib) {}

     void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
       if(printid == ROOT)
         printf("COMM ADDED\n");
       this->sendbuf.push_back(sendbuf);
       this->sendoffset.push_back(sendoffset);
       this->recvbuf.push_back(recvbuf);
       this->recvoffset.push_back(recvoffset);
       this->count.push_back(count);
       this->sendid.push_back(sendid);
       this->recvid.push_back(recvid);
       numcomm++;
     }

     void add(std::vector<T*> inputbuf, T* outputbuf, size_t numreduce, int compid) {
       if(printid == ROOT)
         printf("COMP ADDED\n");
       this->inputbuf.push_back(inputbuf);
       this->outputbuf.push_back(outputbuf);
       this->numreduce.push_back(numreduce);
       this->compid.push_back(compid);
       numcompute++;
     }
  };

  template <typename T>
  class Command {

    public:

    CommBench::Comm<T> *comm = nullptr;
    ExaComm::Compute<T> *compute = nullptr;

    // COMMUNICATION
    Command(CommBench::Comm<T> *comm) : comm(comm) {}
    // COMPUTATION
    Command(ExaComm::Compute<T> *compute) : compute(compute) {}
    // COMMUNICATION + COMPUTATION
    Command(CommBench::Comm<T> *comm, ExaComm::Compute<T> *compute) : comm(comm), compute(compute) {}

    void start() {
      if(comm)
        comm->start();
      else if(compute)
        compute->start();
    }
    void wait() {
      if(comm) {
        comm->wait();
        if(compute)
          compute->run();
      }
      else if(compute)
        compute->wait();
    }
    void run() { start(); wait(); }
    void report() {
      if(comm) {
        if(printid == ROOT) {
          if(compute) printf("COMMAND TYPE: COMMUNICATION + COMPUTATION\n");
          else        printf("COMMAND TYPE: COMMUNICATION\n");
        }
        comm->report();
        if(compute)
          compute->report();
      }
      else if(compute) {
        if(printid == ROOT)
          printf("COMMAND TYPE: COMPUTATION\n");
        compute->report();
      }
    }
    void measure(int warmup, int numiter) {
      if(comm) {
        if(printid == ROOT) {
          if(compute) printf("COMMAND TYPE: COMMUNICATION + COMPUTATION\n");
          else        printf("COMMAND TYPE: COMMUNICATION\n");
        }
	comm->measure(warmup, numiter);
        if(compute)
          compute->measure(warmup, numiter);
      }
      else if(compute) {
        if(printid == ROOT)
          printf("COMMAND TYPE: COMPUTATION\n");
	compute->measure(warmup, numiter);
      }
    }
  };

  template <typename T>
  void implement(std::vector<std::list<Coll<T>*>> &coll_batch, std::vector<std::list<Command<T>>> &pipeline, int pipeoffset) {
    std::vector<int> lib;
    std::vector<int> lib_hash(CommBench::numlib);
    {
      for(int i = 0; i < coll_batch.size(); i++) {
        for(auto &coll : coll_batch[i]){
          if(printid == ROOT) printf("coll->lib: %d\n", coll->lib);
          lib_hash[coll->lib]++;
	}
        for(int j = 0; j < i * pipeoffset; j++)
          coll_batch[i].push_front(new ExaComm::Coll<T>(CommBench::MPI));
      }
     if(printid == ROOT) {
        printf("libraries: ");
        for(int i = 0; i < CommBench::numlib; i++)
          printf("%d ", lib_hash[i]);
        printf("\n");
      }
      for(int i = 0; i < CommBench::numlib; i++)
        if(lib_hash[i]) {
          lib_hash[i] = lib.size();
          lib.push_back(i);
          pipeline.push_back(std::list<Command<T>>());
        }
    }
    {
      using Iter = typename std::list<ExaComm::Coll<T>*>::iterator;
      std::vector<Iter> coll_ptr(coll_batch.size());
      for(int i = 0; i < coll_batch.size(); i++)
        coll_ptr[i] = coll_batch[i].begin();
      while(true) {
        bool finished = true;
        for(int i = 0; i < coll_batch.size(); i++)
          if(coll_ptr[i] != coll_batch[i].end())
            finished = false;
        if(finished)
          break;
        std::vector<CommBench::Comm<T>*> comm_temp(lib.size());
        std::vector<ExaComm::Compute<T>*> compute_temp(lib.size());
        for(int i = 0; i < lib.size(); i++) {
          comm_temp[i] = new CommBench::Comm<T>(CommBench::comm_mpi, (CommBench::library) lib[i]);
          compute_temp[i] = new ExaComm::Compute<T>(CommBench::comm_mpi);
        }
        for(int i = 0; i < coll_batch.size(); i++)
          if(coll_ptr[i] != coll_batch[i].end()) {
            ExaComm::Coll<T> *coll = *coll_ptr[i];
            coll_ptr[i]++;
            for(int i = 0; i < coll->numcomm; i++)
              comm_temp[lib_hash[coll->lib]]->add(coll->sendbuf[i], coll->sendoffset[i], coll->recvbuf[i], coll->recvoffset[i], coll->count[i], coll->sendid[i], coll->recvid[i]);
            for(int i = 0; i < coll->numcompute; i++)
              compute_temp[lib_hash[coll->lib]]->add(coll->inputbuf[i], coll->outputbuf[i], coll->numreduce[i], coll->compid[i]);
          }
        for(int i = 0; i < lib.size(); i++)
          pipeline[i].push_back(Command<T>(comm_temp[i], compute_temp[i]));
      }
    }
  }

#include "src/reduce.h"
#include "src/broadcast.h"

  template <typename T>
  class Comm {

    const MPI_Comm &comm_mpi = CommBench::comm_mpi;

    // PRIMITIVES
    std::vector<std::vector<BROADCAST<T>>> bcast_epoch;
    std::vector<std::vector<REDUCE<T>>> reduce_epoch;
    int numepoch = 0;

    // MACHINE
    std::vector<int> numlevel;
    std::vector<std::vector<int>> groupsize;
    std::vector<std::vector<CommBench::library>> library;

    // FUTURE PIPELINE
    std::vector<std::list<Command<T>>> command_batch;
    std::vector<std::list<Coll<T>*>> coll_batch;

    public:

    void fence() {
      bcast_epoch.push_back(std::vector<BROADCAST<T>>());
      reduce_epoch.push_back(std::vector<REDUCE<T>>());
      numepoch++;
    }

    Comm(const MPI_Comm &comm_mpi_temp) {
      // INITIALIZE COMMBENCH
#ifdef CAP_NCCL
      CommBench::Comm<T> comm(comm_mpi_temp, CommBench::NCCL);
#else
      CommBench::Comm<T> comm(comm_mpi_temp, CommBench::MPI);
#endif
      MPI_Comm_rank(comm_mpi_temp, &printid);
      // DEFAULT EPOCH
      fence();
    }

    // ADD FUNCTIONS FOR BROADCAST AND REDUCE PRIMITIVES
    void add_bcast(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
      bcast_epoch[numepoch-1].push_back(BROADCAST<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvid));
    }
    void add_bcast(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, std::vector<int> &recvids) {
      bcast_epoch[numepoch-1].push_back(BROADCAST<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvids));
    }
    void add_reduce(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
      reduce_epoch[numepoch-1].push_back(REDUCE<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvid));
    }
    void add_reduce(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, std::vector<int> &sendids, int recvid) {
      reduce_epoch[numepoch-1].push_back(REDUCE<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendids, recvid));
    }

    // INITIALIZE BROADCAST AND REDUCTION TREES
    void init(int numlevel, int groupsize[], CommBench::library lib[], int numstripe, int stripeoffset, int numbatch, int pipelineoffset) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);

      if(printid == ROOT) {
        printf("NUMBER OF EPOCHS: %d\n", numepoch);
        for(int epoch = 0; epoch < numepoch; epoch++)
          printf("epoch %d: %zu bcast %zu reduction\n", epoch, bcast_epoch[epoch].size(), reduce_epoch[epoch].size());
        printf("Initialize ExaComm with %d levels\n", numlevel);
        for(int level = 0; level < numlevel; level++) {
          printf("level %d groupsize %d library: ", level, groupsize[level]);
          switch(lib[level]) {
            case CommBench::IPC  : printf("IPC"); break;
            case CommBench::MPI  : printf("MPI"); break;
            case CommBench::NCCL : printf("NCCL"); break;
            default : break;
          }
          if(level == 0)
            if(groupsize[0] != numproc)
              printf(" *");
          printf("\n");
        }
        printf("\n");
      }

      // ALLOCATE COMMAND BATCH
      for(int batch = 0; batch < numbatch; batch++)
        coll_batch.push_back(std::list<Coll<T>*>());

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
          for(int batch = 0; batch < numbatch; batch++) {
            // STRIPE REDUCTION
            std::vector<BROADCAST<T>> merge_list;
            ExaComm::stripe(comm_mpi, numstripe, stripeoffset, reduce_batch[batch], merge_list);
            // HIERARCHICAL REDUCTION RING + TREE
	    std::vector<REDUCE<T>> reduce_intra; // for accumulating intra-node communications for tree (internally)
            ExaComm::reduce_ring(comm_mpi, numlevel, groupsize, lib, reduce_batch[batch], reduce_intra, coll_batch[batch]);
            // COMPLETE STRIPING BY INTRA-NODE GATHER
            std::vector<int> groupsize_temp(groupsize, groupsize + numlevel);
            groupsize_temp[0] = numproc;
            ExaComm::bcast_tree(comm_mpi, numlevel, groupsize_temp.data(), lib, merge_list, 1, coll_batch[batch]);
	  }
        }
        // INIT BROADCAST
        std::vector<BROADCAST<T>> &bcastlist = bcast_epoch[epoch];
        if(bcastlist.size())
        {
          // PARTITION INTO BATCHES
          std::vector<std::vector<BROADCAST<T>>> bcast_batch(numbatch);
          ExaComm::batch(bcastlist, numbatch, bcast_batch);
          // FOR EACH BATCH
          for(int batch = 0; batch < numbatch; batch++) {
            // STRIPE BROADCAST
            std::vector<REDUCE<T>> split_list;
	    ExaComm::stripe(comm_mpi, numstripe, stripeoffset, bcast_batch[batch], split_list);
            std::vector<int> groupsize_temp(groupsize, groupsize + numlevel);
            groupsize_temp[0] = numproc;
            // INITIALIZE STRIPING BY INTRA-NODE SCATTER
            std::vector<T*> recvbuff; // for memory recycling
            ExaComm::reduce_tree(comm_mpi, numlevel, groupsize_temp.data(), lib, split_list, numlevel - 1, coll_batch[batch], recvbuff, 0);
            // HIERARCHICAL RING + TREE
            std::vector<BROADCAST<T>> bcast_intra; // for accumulating intra-node communications for tree (internally)
            ExaComm::bcast_ring(comm_mpi, numlevel, groupsize, lib, bcast_batch[batch], bcast_intra, coll_batch[batch]);
          }
        }
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
          printf("coll_batch size %zu: ", coll_batch.size());
          for(int i = 0; i < coll_batch.size(); i++)
            printf("%zu ", coll_batch[i].size());
          printf("\n\n");
        }
      }

      implement(coll_batch, command_batch, pipelineoffset);

      if(printid == ROOT) {
        printf("********************************************\n\n");
        printf("pipeline depth %zu\n", coll_batch.size());
        printf("coll_list size %zu\n", coll_batch[0].size());
        printf("\n");
      }
      {
        int print_batch_size = (coll_batch.size() > 16 ? 16 : coll_batch.size());
        using Iter = typename std::list<ExaComm::Coll<T>*>::iterator;
        std::vector<Iter> coll_ptr(print_batch_size);
        for(int i = 0; i < print_batch_size; i++)
          coll_ptr[i] = coll_batch[i].begin();
        int collindex = 0;
        while(true) {
          if(printid == ROOT)
            printf("proc %d collindex %d: |", printid, collindex);
          bool finished = true;
          for(int i = 0; i < print_batch_size; i++) {
            if(coll_ptr[i] != coll_batch[i].end()) {
              if((*coll_ptr[i])->numcomm) {
                if(printid == ROOT) {
                  printf(" %d", (*coll_ptr[i])->numcomm);
                  switch((*coll_ptr[i])->lib) {
                    case CommBench::IPC :  printf(" IPC"); break;
                    case CommBench::MPI :  printf(" MPI"); break;
                    case CommBench::NCCL : printf(" NCL"); break;
                    default : break;
                  }
                }
                if((*coll_ptr[i])->numcompute) {
                  if(printid == ROOT)
                    printf(" %d* |", (*coll_ptr[i])->numcompute);
                }
                else
                  if(printid == ROOT)
                    printf("    |");
              }
              else if((*coll_ptr[i])->numcompute) {
                //MPI_Allreduce(MPI_IN_PLACE, &numcomp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                if(printid == ROOT)
                  printf("  %d  *** |", (*coll_ptr[i])->numcompute);
              }
              else
                if(printid == ROOT)
                  printf("     -     |");
              finished = false;
              coll_ptr[i]++;
            }
            else
              if(printid == ROOT)
                printf("          |");
          }
          if(printid == ROOT)
            printf("\n");
          if(finished)
            break;
          collindex++;
        }
      }
    };


    void run() {
      using Iter = typename std::list<ExaComm::Command<T>>::iterator;
      std::vector<Iter> commandptr(command_batch.size());
      for(int i = 0; i < command_batch.size(); i++)
        commandptr[i] = command_batch[i].begin();
      while(true) {
        bool finished = true;
        for(int i = 0; i < command_batch.size(); i++)
          if(commandptr[i] != command_batch[i].end()) {
            commandptr[i]->start();
            finished = false;
          }
        if(finished)
          break;
        for(int i = command_batch.size() - 1; i > -1; i--)
          if(commandptr[i] != command_batch[i].end()) {
            commandptr[i]->wait();
            commandptr[i]++;
          }
      }
    }

    void measure(int warmup, int numiter) {
      if(printid == ROOT) {
        printf("command_batch size %zu\n", command_batch.size());
        printf("commandlist size %zu\n", command_batch[0].size());
      }
      {
        using Iter = typename std::list<ExaComm::Command<T>>::iterator;
        std::vector<Iter> commandptr(command_batch.size());
        for(int i = 0; i < command_batch.size(); i++)
          commandptr[i] = command_batch[i].begin();
        while(true) {
          bool finished = true;
          for(int i = 0; i < command_batch.size(); i++)
            if(commandptr[i] != command_batch[i].end())
              finished = false;
          if(finished)
            break;
          if(printid == ROOT) printf("******************************************* MEASURE COMMANDS ************************************************\n");
          for(int i = 0; i < command_batch.size(); i++)
            if(commandptr[i] != command_batch[i].end())
              commandptr[i]->measure(warmup, numiter);
          if(printid == ROOT) printf("******************************************* MEASURE STEP ************************************************\n");
          MPI_Barrier(comm_mpi);
          double time = MPI_Wtime();
          for(int i = 0; i < command_batch.size(); i++)
            if(commandptr[i] != command_batch[i].end())
              commandptr[i]->start();
          for(int i = 0; i < command_batch.size(); i++)
            if(commandptr[i] != command_batch[i].end()) {
              commandptr[i]->wait();
              commandptr[i]++;
            }
          MPI_Barrier(comm_mpi);
          time = MPI_Wtime() - time;
          if(printid == ROOT)
            printf("time: %e\n", time);
        }
      }
    }

    void report() {
      if(printid == ROOT) {
        printf("command_batch size %zu\n", command_batch.size());
        printf("commandlist size %zu\n", command_batch[0].size());
      }
      int command = 0;
      for(auto it = command_batch[0].begin(); it != command_batch[0].end(); it++) {
        if(printid == ROOT)
          printf("command %d", command);
        it->report();
        command++;
      }
    }

    void time() {
      if(printid == ROOT) {
        printf("********************************************\n\n");
        printf("pipeline depth %zu\n", command_batch.size());
        printf("commandlist size %zu\n", command_batch[0].size());
        printf("\n");
      }
      int print_batch_size = (command_batch.size() > 16 ? 16 : command_batch.size());
      {
        using Iter = typename std::list<ExaComm::Command<T>>::iterator;
        std::vector<Iter> commandptr(print_batch_size);
        for(int i = 0; i < print_batch_size; i++)
          commandptr[i] = command_batch[i].begin();
        int command = 0;
        while(true) {
          if(printid == ROOT)
            printf("proc %d command %d: |", printid, command);
          bool finished = true;
          for(int i = 0; i < print_batch_size; i++) {
            if(commandptr[i] != command_batch[i].end()) {
              if(commandptr[i]->comm) {
                int numsend = commandptr[i]->comm->numsend;
                int numrecv = commandptr[i]->comm->numrecv;
                //MPI_Allreduce(MPI_IN_PLACE, &numsend, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                //MPI_Allreduce(MPI_IN_PLACE, &numrecv, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                if(printid == ROOT) {
                  if(numsend) printf(" %d", numsend);
                  else        printf("  ");
                  if(numrecv) printf("+%d", numrecv);
                  else        printf("  ");
                  if(numsend+numrecv)
                    switch(commandptr[i]->comm->lib) {
                      case CommBench::IPC :  printf(" IPC"); break;
                      case CommBench::MPI :  printf(" MPI"); break;
                      case CommBench::NCCL : printf(" NCL"); break;
                      default : break;
                    }
                  else // printf("-   ");
                    switch(commandptr[i]->comm->lib) {
                      case CommBench::IPC :  printf("I   "); break;
                      case CommBench::MPI :  printf("M   "); break;
                      case CommBench::NCCL : printf("N   "); break;
                      default : break;
                    }
                }
                if(commandptr[i]->compute) {
                  int numcomp = commandptr[i]->compute->numcomp;
                  //MPI_Allreduce(MPI_IN_PLACE, &numcomp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                  if(printid == ROOT) {
                    if(numcomp) printf(" %d*", numcomp);
                    else        printf("*  ");
                  }
                }
                if(printid == ROOT)
                  printf(" |");
	      }
	      else if(commandptr[i]->compute) {
                int numcomp = commandptr[i]->compute->numcomp;
                //MPI_Allreduce(MPI_IN_PLACE, &numcomp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                if(printid == ROOT) {
                  if(numcomp) printf("  %d  *** |", numcomp);
                  else        printf("    *    |");
                }
              }
              finished = false;
              commandptr[i]++;
            }
            else
              if(printid == ROOT)
                printf("         |");
          }
          if(printid == ROOT)
            printf("\n");
          if(finished)
            break;
          command++;
        }
      }

      using Iter = typename std::list<ExaComm::Command<T>>::iterator;
      std::vector<Iter> commandptr(command_batch.size());
      for(int i = 0; i < command_batch.size(); i++)
        commandptr[i] = command_batch[i].begin();

      int command = 0;
      double totalstarttime = 0;
      double totalwaittime = 0;
      MPI_Barrier(comm_mpi);
      double totaltime = MPI_Wtime();
      while(true) {
        double starttime;
        double waittime;
        bool finished = true;
        {
          MPI_Barrier(comm_mpi);
          double time = MPI_Wtime();
          for(int i = 0; i < command_batch.size(); i++)
            if(commandptr[i] != command_batch[i].end()) {
              commandptr[i]->start();
              finished = false;
            }
          MPI_Barrier(comm_mpi);
          starttime = MPI_Wtime() - time;
        }
        MPI_Allreduce(MPI_IN_PLACE, &finished, 1, MPI_C_BOOL, MPI_LOR, comm_mpi);
        if(finished)
          break;
        {
          MPI_Barrier(comm_mpi);
          double time = MPI_Wtime();
          for(int i = 0; i < command_batch.size(); i++)
            if(commandptr[i] != command_batch[i].end()) {
              commandptr[i]->wait();
              commandptr[i]++;
            }
          MPI_Barrier(comm_mpi);
          waittime = MPI_Wtime() - time;
        }
        if(printid == ROOT)
          printf("command %d start: %e wait: %e\n", command, starttime, waittime);
        totalstarttime += starttime;
        totalwaittime += waittime;
        command++;
      }
      MPI_Barrier(comm_mpi);
      totaltime = MPI_Wtime() - totaltime;
      if(printid == ROOT) {
        printf("start %e wait %e other %e\n", totalstarttime, totalwaittime, totaltime - totalstarttime - totalwaittime); 
        printf("total time %e\n", totaltime);
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
