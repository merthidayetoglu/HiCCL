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

namespace ExaComm {

  int printid;
  FILE *pFile;
  size_t buffsize = 0;

  enum command {start, wait, run};
  enum pattern {pt2pt, gather, scatter, reduce, broadcast, alltoall, allreduce, allgather, reducescatter};

  template <typename T>
  struct Command {
    public:
    CommBench::Comm<T> *comm;
    command com;
    Command(CommBench::Comm<T> *comm, command com) : comm(comm), com(com) {
      if(printid == ROOT) {
        switch(com) {
          case(command::start) : printf("command::start added\n"); break;
          case(command::wait) : printf("command::wait added\n"); break;
          case(command::run) : printf("command::run added\n"); break;
        }
      }
    }
  };

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

  template <typename T>
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
  }

#if defined PORT_CUDA || defined PORT_HIP
  template <typename T>
  __global__ void reduce_kernel(T* outputbuf, int numinput, T** inputbuf, size_t count) {
     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i < count) {
       T output = 0;
       for(int input = 0; input = numinput; input++)
         output += inputbuf[input][i];
       outputbuf[i] = output;
     }
  }
#else
  template <typename T>
    void reduce_kernel(T* outputbuf, int numinput, T** inputbuf, size_t count) {
    #pragma omp parallel for
    for(size_t i = 0; i < count; i++) {
      T output = 0;
      for(int input = 0; input < numinput; input++)
        output += inputbuf[input][i];
      outputbuf[i] = output;
    }
  }
#endif

  template <typename T>
  class Comp {

    MPI_Comm comm_mpi;
    int numcomp = 0;
	
    std::vector<std::vector<T*>> inputbuf;
    std::vector<T*> outputbuf;
    std::vector<size_t> count;
#ifdef PORT_CUDA
    std::vector<cudaStream_t*> stream;
#elif defined PORT_HIP
    std::vector<cudaStream_t*> stream;
#endif

    public:

    Comp(const MPI_Comm &comm_mpi_temp) {
      MPI_Comm_dup(comm_mpi_temp, &comm_mpi); // CREATE SEPARATE COMMUNICATOR EXPLICITLY
    }

    void add(std::vector<T*> &inputbuf, T *outputbuf, size_t count, int compid) {
      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);
      if(myid == compid) {
        this->inputbuf.push_back(inputbuf);
        this->outputbuf.push_back(outputbuf);
        this->count.push_back(count);
#ifdef PORT_CUDA
        stream.push_back(new cudaStream_t);
        cudaStreamCreate(stream[numcomp]);
#elif defined PORT_HIP
        stream.push_back(new hipStream_t);
        hipStreamCreate(stream[numcomp]);
#endif
        numcomp++;
      }
    }

    void start() {
      int blocksize = 256;
      for(int comp = 0; comp < numcomp; comp++) {
#if defined PORT_CUDA || PORT_HIP
        int numblocks = (count[comp] + blocksize - 1) / blocksize;
        reduce_kernel<<<blocksize, numblocks, 0, *stream[comp]>>> (outputbuf[comp], inputbuf[comp].size(), inputbuf[comp].data(), count[comp]);
#else
        reduce_kernel(outputbuf[comp], inputbuf[comp].size(), inputbuf[comp].data(), count[comp]);
#endif
      } 
    }

    void wait() {
      for(int comp = 0; comp < numcomp; comp++)
#ifdef PORT_CUDA
        cudaStreamSynchronize(*stream[comp]);
#elif defined PORT_HIP
        hipStreamSynchronize(*stream[comp]);
#endif
    }

    void run() { start(); wait(); }
  };

#include "bcast.h"
#include "reduce.h"

  template <typename T>
  class Comm {

    const MPI_Comm comm_mpi;

    std::vector<BCAST<T>> bcastlist;
    std::vector<REDUCE<T>> reducelist;

    std::list<CommBench::Comm<T>*> commlist;
    std::list<Command<T>> commandlist;
    std::list<T*> bufferlist;

    // PIPELINING
    std::vector<std::list<CommBench::Comm<T>*>> comm_batch;

    public:

    Comm(const MPI_Comm &comm_mpi) : comm_mpi(comm_mpi) {}

    // ADD FUNCTIONS FOR BCAST AND REDUCE PRIMITIVES
    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
      std::vector<int> recvids = {recvid};
      bcastlist.push_back(BCAST<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvids));
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
        for(auto &bcast : bcastlist) {
          int batchsize = bcast.count / numbatch;
          for(int batch = 0; batch < numbatch; batch++)
            bcast_batch[batch].push_back(BCAST<T>(bcast.sendbuf, bcast.sendoffset + batch * batchsize, bcast.recvbuf, bcast.recvoffset + batch * batchsize, batchsize, bcast.sendid, bcast.recvid));
        }
        // STRIPE BROADCAST
        std::vector<std::list<CommBench::Comm<T>*>> comm_batch(numbatch);
	for(int batch = 0; batch < numbatch; batch++) {
          std::list<Command<T>> commandlist;
          striped(comm_mpi, numlevel, groupsize, lib, bcast_batch[batch], comm_batch[batch], commandlist);
        }
        this->comm_batch = comm_batch;
      }
      // INIT REDUCE
      if(reducelist.size()) {
        // PARTITION REDUCTION INTO BATCHES
        std::vector<std::vector<REDUCE<T>>> reduce_batch(numbatch);
        for(auto &reduce : reducelist) {
          int batchsize = reduce.count / numbatch;
          for(int batch = 0; batch < numbatch; batch++)
            reduce_batch[batch].push_back(REDUCE<T>(reduce.sendbuf, reduce.sendoffset + batch * batchsize, reduce.recvbuf, reduce.recvoffset + batch * batchsize, batchsize, reduce.sendid, reduce.recvid));
        }
        // STRIPE REDUCE
        std::vector<std::list<CommBench::Comm<T>*>> comm_batch(numbatch);
        for(int batch = 0; batch < numbatch; batch++) {
          std::list<Command<T>> commandlist;
          // striped(comm_mpi, numlevel, groupsize, lib, reduce_batch[batch], comm_batch[batch], commandlist);
        }
        this->comm_batch = comm_batch;
      }
      // ADD INITIAL DUMMY COMMUNICATORS INTO THE PIPELINE
      if(bcastlist.size() | reducelist.size()) {
        for(int batch = 0; batch < numbatch; batch++)
          for(int c = 0; c < batch; c++)
            comm_batch[batch].push_front(new CommBench::Comm<T>(comm_mpi, CommBench::MPI));
        // REPORT MEMORY USAGE
        std::vector<size_t> buffsize_all(numproc);
        MPI_Allgather(&buffsize, sizeof(size_t), MPI_BYTE, buffsize_all.data(), sizeof(size_t), MPI_BYTE, comm_mpi);
        if(myid == ROOT)
          for(int p = 0; p < numproc; p++)
            printf("ExaComm Memory [%d]: %zu bytes\n", p, buffsize_all[p] * sizeof(size_t));
      }
    };

    void overlap_batch() {

      using Iter = typename std::list<CommBench::Comm<T>*>::iterator;
      std::vector<Iter> commptr(comm_batch.size());
      for(int i = 0; i < comm_batch.size(); i++)
        commptr[i] = comm_batch[i].begin();

      bool finished = false;
      while(!finished) {
        finished = true;
        for(int i = 0; i < comm_batch.size(); i++)
          if(commptr[i] != comm_batch[i].end())
            (*commptr[i])->start();
        for(int i = 0; i < comm_batch.size(); i++)
          if(commptr[i] != comm_batch[i].end())
            (*commptr[i])->wait();
        for(int i = 0; i < comm_batch.size(); i++)
          if(commptr[i] != comm_batch[i].end()) {
            commptr[i]++;
            finished = false;
          }
      }
    }

    void run_batch() {
      for(auto list : comm_batch)
        for(auto comm : list)
          comm->run();
    }

    void run_commlist() {
      for(auto comm : commlist)
        comm->run();
    }

    void run_commandlist() {
      for(auto &command : commandlist)
        switch(command.com) {
          case(ExaComm::start) : command.comm->start(); break;
          case(ExaComm::wait) : command.comm->wait(); break;
          case(ExaComm::run) : command.comm->run(); break;
        }
    }

    void measure(int warmup, int numiter) {
      for(auto comm : commlist)
        comm->measure(warmup, numiter);
      if(printid == ROOT) {
        printf("commlist size %zu\n", commlist.size());
        printf("commandlist size %zu\n", commandlist.size());
        printf("bufferlist size %zu\n", bufferlist.size());
      }
      for(auto &list : comm_batch)
        for(auto comm : list)
          comm->measure(warmup, numiter);
      if(printid == ROOT) {
        printf("comm_batch size %zu: ", comm_batch.size());
        for(int i = 0; i < comm_batch.size(); i++)
          printf("%zu ", comm_batch[i].size());
        printf("\n");
      }
    }

    void report() {
      int counter = 0;
      for(auto it = commandlist.begin(); it != commandlist.end(); it++) {
        if(printid == ROOT) {
          printf("counter: %d command::", counter);
          switch(it->com) {
            case(ExaComm::start) : printf("start\n"); break;
            case(ExaComm::wait) : printf("wait\n"); break;
            case(ExaComm::run) : printf("run\n"); break;
          }
        }
        it->comm->report();
        counter++;
      }
      if(printid == ROOT) {
        printf("commandlist size %zu\n", commandlist.size());
        printf("commlist size %zu\n", commlist.size());
        printf("bufferlist size %zu\n", bufferlist.size());
      }
    }
  };

#include <unistd.h>

#include "bench.h"

}
