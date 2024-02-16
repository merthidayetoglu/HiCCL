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

  template <typename T>
  class Comm {

    // PRIMITIVES
    std::vector<std::vector<BROADCAST<T>>> bcast_epoch;
    std::vector<std::vector<REDUCE<T>>> reduce_epoch;
    int numepoch = 0;

    public:

    // HiCCL PARAMETERS
    std::vector<int> hierarchy = {numproc};
    std::vector<CommBench::library> library = {CommBench::MPI};
    int numstripe = 1;
    int ringnodes = 1;
    int pipedepth = 1;
    // ENDPOINTS
    T *sendbuf = nullptr;
    T *recvbuf = nullptr;
    size_t sendcount = 0;
    size_t recvcount = 0;

    // SET HIERARCHY
    void set_hierarchy(std::vector<int> hierarchy, std::vector<CommBench::library> library) {
      if(hierarchy.size() != library.size()) {
        if(myid == printid)
          printf("hierarchy and library must have the same size!\n");
        return;
      }
      else {
        this->hierarchy = hierarchy;
        this->library = library;
      }
    }

    // SET ENDPOINTS
    void set_endpoints(T *sendbuf, size_t sendcount, T *recvbuf, size_t recvcount) {
      this->sendbuf = sendbuf;
      this->sendcount = sendcount;
      this->recvbuf = recvbuf;
      this->recvcount = recvcount;
    }

    void print_parameters() {
      if(myid == printid) {
        printf("**************** HiCCL PARAMETERS\n");
        printf("%ld-level hierarchy:\n", hierarchy.size());
        for(int i = 0; i < hierarchy.size(); i++) {
          printf("  level %d factor: %d library: ", i, hierarchy[i]);
          CommBench::print_lib(library[i]);
	  if(hierarchy[0] == numproc && library[0] == CommBench::MPI)
            printf(" (default)\n");
          else
            printf("\n");
        }
        printf("numstripe: %d", numstripe);
        if(numstripe == 1)
          printf(" (default)\n");
        else
          printf("\n");
        printf("ringnodes: %d", ringnodes);
        if(ringnodes == 1)
          printf(" (default)\n");
        else
          printf("\n");
        printf("pipedepth: %d", pipedepth);
        if(pipedepth == 1)
          printf(" (default)\n");
        else
          printf("\n");
        printf("sendbuf: %p, sendcount %ld", sendbuf, sendcount);
        if(sendbuf == nullptr)
          printf(" (default)\n");
        else
          printf("\n");
        printf("recvbuf: %p, recvcount %ld", recvbuf, recvcount);
        if(recvbuf == nullptr)
          printf(" (default)\n");
        else
          printf("\n");
        printf("*********************************\n");
      }
    }

    // FUTURE PIPELINE
    std::vector<std::list<Command<T>>> command_batch;
    std::vector<std::list<Coll<T>*>> coll_batch;

    void add_fence() {
      bcast_epoch.push_back(std::vector<BROADCAST<T>>());
      reduce_epoch.push_back(std::vector<REDUCE<T>>());
      if(myid == printid)
        printf("Add epoch %d\n", numepoch);
      numepoch++;
    }

    Comm() {
      // DEFAULT EPOCH
      add_fence();
      // DEFAULT PARAMETERS
      if(myid == printid) {
        printf("DEFAULT PARAMETERS:\n");
        print_parameters();
      }
    }

    // ADD FUNCTIONS FOR BROADCAST AND REDUCE PRIMITIVES
    void add_bcast(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, std::vector<int> &recvids) {
      bcast_epoch.back().push_back(BROADCAST<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvids));
      bcast_epoch.back().back().report();
    }
    void add_bcast(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
      bcast_epoch.back().push_back(BROADCAST<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvid));
      bcast_epoch.back().back().report();
    }
    void add_bcast(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, pattern recv_pattern) {
      int recvid = (recv_pattern == pattern::others ? -1 : numproc);
      bcast_epoch.back().push_back(BROADCAST<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvid));
      bcast_epoch.back().back().report();
    }
    void add_reduce(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, std::vector<int> &sendids, int recvid) {
      reduce_epoch.back().push_back(REDUCE<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendids, recvid));
      reduce_epoch.back().back().report();
    }
    void add_reduce(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
      reduce_epoch.back().push_back(REDUCE<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvid));
      reduce_epoch.back().back().report();
    }
    void add_reduce(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, pattern send_pattern, int recvid) {
      int sendid = (send_pattern == pattern::others ? -1 : numproc);
      reduce_epoch.back().push_back(REDUCE<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvid));
      reduce_epoch.back().back().report();
    }

#include "init.h"

    void init() {
      if(myid == printid) {
        printf("FINAL PARAMETERS\n");
        print_parameters();
      }
      // CONVERT FACTORIZATION TO GROUPSIZE
      int numlevel = hierarchy.size();
      std::vector<int> groupsize(numlevel);
      groupsize[numlevel - 1] = hierarchy[numlevel - 1];
      for(int i = numlevel - 2; i > -1; i--)
        groupsize[i] = groupsize[i + 1] * hierarchy[i];
      MPI_Barrier(comm_mpi);
      double init_time = MPI_Wtime();
      // init.h
      init(numlevel, groupsize.data(), library.data(), numstripe, pipedepth);
      MPI_Barrier(comm_mpi);
      if(myid == printid)
        printf("initialization time: %e seconds\n", MPI_Wtime() - init_time);
    }

    void run() {
      //printf("command_batch size %ld\n", command_batch.size());
      //for(auto list : command_batch)
      //  printf("list size: %ld\n", list.size());
      using Iter = typename std::list<Command<T>>::iterator;
      std::vector<Iter> commandptr(command_batch.size());
      for(int i = 0; i < command_batch.size(); i++)
        commandptr[i] = command_batch[i].begin();
      while(true) {
        bool finished = true;
        for(int i = 0; i < command_batch.size(); i++)
          if(commandptr[i] != command_batch[i].end()) {
            commandptr[i]->comm->start();
            finished = false;
          }
        if(finished)
          break;
        for(int i = command_batch.size() - 1; i > -1; i--)
          if(commandptr[i] != command_batch[i].end()) {
            commandptr[i]->comm->wait();
            commandptr[i]->compute->start();
          }
        for(int i = 0; i < command_batch.size(); i++)
          if(commandptr[i] != command_batch[i].end()) {
            commandptr[i]->compute->wait();
            commandptr[i]++;
          }
      }
    }

    void run(T *sendbuf, T *recvbuf) {
      CommBench::memcpyD2D(this->sendbuf, sendbuf, sendcount);
      run();
      CommBench::memcpyD2D(recvbuf, this->recvbuf, recvcount);
    }

    void measure(int warmup, int numiter, size_t count) {
      if(myid == printid) {
        printf("command_batch size %zu\n", command_batch.size());
        if(command_batch.size())
          printf("commandlist size %zu\n", command_batch[0].size());
      }
      MPI_Barrier(comm_mpi);
      {
        using Iter = typename std::list<Command<T>>::iterator;
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
          if(myid == printid) printf("******************************************* MEASURE COMMANDS ************************************************\n");
          for(int i = 0; i < command_batch.size(); i++)
            if(commandptr[i] != command_batch[i].end()) {
              commandptr[i]->measure(warmup, numiter, count);
              commandptr[i]++;
            }
          /*if(myid == printid) printf("******************************************* MEASURE STEP ************************************************\n");
          MPI_Barrier(comm_mpi);
          double time = MPI_Wtime();
          for(int i = 0; i < command_batch.size(); i++)
            if(commandptr[i] != command_batch[i].end())
              commandptr[i]->comm->start();
          for(int i = 0; i < command_batch.size(); i++)
            if(commandptr[i] != command_batch[i].end()) {
              commandptr[i]->comm->wait();
              commandptr[i]++;
            }
          MPI_Barrier(comm_mpi);
          time = MPI_Wtime() - time;
          if(myid == printid)
            printf("time: %e\n", time);*/
        }
      }
    }

    void report() {
      if(myid == printid) {
        printf("command_batch size %zu\n", command_batch.size());
        printf("commandlist size %zu\n", command_batch[0].size());
      }
      int command = 0;
      for(auto it = command_batch[0].begin(); it != command_batch[0].end(); it++) {
        if(myid == printid)
          printf("command %d", command);
        it->report();
        command++;
      }
    }

    void time() {
      if(myid == printid) {
        printf("********************************************\n\n");
        printf("pipeline depth %zu\n", command_batch.size());
        printf("commandlist size %zu\n", command_batch[0].size());
        printf("\n");
      }
      int print_batch_size = (command_batch.size() > 16 ? 16 : command_batch.size());
      {
        using Iter = typename std::list<Command<T>>::iterator;
        std::vector<Iter> commandptr(print_batch_size);
        for(int i = 0; i < print_batch_size; i++)
          commandptr[i] = command_batch[i].begin();
        int command = 0;
        while(true) {
          if(myid == printid)
            printf("proc %d command %d: |", myid, command);
          bool finished = true;
          for(int i = 0; i < print_batch_size; i++) {
            if(commandptr[i] != command_batch[i].end()) {
              if(commandptr[i]->comm) {
                int numsend = commandptr[i]->comm->numsend;
                int numrecv = commandptr[i]->comm->numrecv;
                //MPI_Allreduce(MPI_IN_PLACE, &numsend, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                //MPI_Allreduce(MPI_IN_PLACE, &numrecv, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                if(myid == printid) {
                  if(numsend) printf(" %d", numsend);
                  else        printf("  ");
                  if(numrecv) printf("+%d", numrecv);
                  else        printf("  ");
                  if(numsend+numrecv)
                    switch(commandptr[i]->comm->lib) {
                      case CommBench::IPC :  printf(" IPC"); break;
                      case CommBench::MPI :  printf(" MPI"); break;
		      case CommBench::XCCL : printf(" XCCL"); break;
                      default : break;
                    }
                  else // printf("-   ");
                    switch(commandptr[i]->comm->lib) {
                      case CommBench::IPC  : printf("I   "); break;
                      case CommBench::MPI  : printf("M   "); break;
                      case CommBench::XCCL : printf("X   "); break;
                      default : break;
                    }
                }
                if(commandptr[i]->compute) {
                  int numcomp = commandptr[i]->compute->numcomp;
                  //MPI_Allreduce(MPI_IN_PLACE, &numcomp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                  if(myid == printid) {
                    if(numcomp) printf(" %d*", numcomp);
                    else        printf("*  ");
                  }
                }
                if(myid == printid)
                  printf(" |");
	      }
	      else if(commandptr[i]->compute) {
                int numcomp = commandptr[i]->compute->numcomp;
                //MPI_Allreduce(MPI_IN_PLACE, &numcomp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                if(myid == printid) {
                  if(numcomp) printf("  %d  *** |", numcomp);
                  else        printf("    *    |");
                }
              }
              finished = false;
              commandptr[i]++;
            }
            else
              if(myid == printid)
                printf("         |");
          }
          if(myid == printid)
            printf("\n");
          if(finished)
            break;
          command++;
        }
      }

      using Iter = typename std::list<Command<T>>::iterator;
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
        if(myid == printid)
          printf("command %d start: %e wait: %e\n", command, starttime, waittime);
        totalstarttime += starttime;
        totalwaittime += waittime;
        command++;
      }
      MPI_Barrier(comm_mpi);
      totaltime = MPI_Wtime() - totaltime;
      if(myid == printid) {
        printf("start %e wait %e other %e\n", totalstarttime, totalwaittime, totaltime - totalstarttime - totalwaittime); 
        printf("total time %e\n", totaltime);
      }
    }
  };
