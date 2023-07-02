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
            // (*commptr[i])->wait();
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

  template <typename T>
  struct P2P {
    public:
    T* const sendbuf;
    const size_t sendoffset;
    T* const recvbuf;
    size_t const recvoffset;
    size_t const count;
    int const sendid;
    int const recvid;

    P2P(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid)
    : sendbuf(sendbuf), sendoffset(sendoffset), recvbuf(recvbuf), recvoffset(recvoffset), count(count), sendid(sendid), recvid(recvid) {}
  };

  template <typename T>
  struct BCAST {
    public:
    T* const sendbuf;
    const size_t sendoffset;
    T* const recvbuf;
    const size_t recvoffset;
    const size_t count;
    const int sendid;
    std::vector<int> recvid;

    BCAST(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, std::vector<int> &recvid)
    : sendbuf(sendbuf), sendoffset(sendoffset), recvbuf(recvbuf), recvoffset(recvoffset), count(count), sendid(sendid), recvid(recvid) {}
    void report(int id) {
      if(printid == sendid) {
        MPI_Send(&sendbuf, sizeof(T*), MPI_BYTE, id, 0, MPI_COMM_WORLD);
        MPI_Send(&sendoffset, sizeof(size_t), MPI_BYTE, id, 0, MPI_COMM_WORLD);
      }
      for(auto recvid : this->recvid)
        if(printid == recvid) {
          MPI_Send(&recvbuf, sizeof(T*), MPI_BYTE, id, 0, MPI_COMM_WORLD);
          MPI_Send(&recvoffset, sizeof(size_t), MPI_BYTE, id, 0, MPI_COMM_WORLD);
        }
      if(printid == id) {
        T* sendbuf_sendid;
        T* recvbuf_recvid[recvid.size()];
        size_t sendoffset_sendid;
        size_t recvoffset_recvid[recvid.size()];
        MPI_Recv(&sendbuf_sendid, sizeof(T*), MPI_BYTE, sendid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&sendoffset_sendid, sizeof(size_t), MPI_BYTE, sendid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for(int recv = 0; recv < recvid.size(); recv++) {
          MPI_Recv(recvbuf_recvid + recv, sizeof(T*), MPI_BYTE, recvid[recv], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(recvoffset_recvid + recv, sizeof(size_t), MPI_BYTE, recvid[recv], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        printf("BCAST report: count %lu\n", count);
        char text[1000];
        int n = sprintf(text, "sendid %d sendbuf %p sendoffset %lu -> ", sendid, sendbuf_sendid, sendoffset_sendid);
        printf("%s", text);
        memset(text, ' ', n);
        for(int recv = 0; recv < recvid.size(); recv++) {
          printf("recvid: %d recvbuf %p recvoffset %lu\n", recvid[recv], recvbuf_recvid[recv], recvoffset_recvid[recv]);
          printf("%s", text);
        }
        printf("\n");
      }
    }
  };

// #define FACTOR_LEVEL
  #define FACTOR_LOCAL
  template <typename T>
  void bcast_tree(const MPI_Comm &comm_mpi, int numlevel, int groupsize[], CommBench::library lib[], std::vector<BCAST<T>> bcastlist, std::list<CommBench::Comm<T>*> &commlist, int level, std::list<Command<T>> &commandlist, std::list<Command<T>> &waitlist, int nodelevel) {

    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    if(numproc != groupsize[0]) {
      printf("ERROR!!! groupsize[0] must be equal to numproc.\n");
      return;
    }
    if(bcastlist.size() == 0)
      return;

    CommBench::Comm<T> *comm_temp = new CommBench::Comm<T>(comm_mpi, lib[level-1]);
    commlist.push_back(comm_temp);

#ifdef FACTOR_LOCAL
    if(level > nodelevel)
      commandlist.push_back(Command<T>(comm_temp, command::start));
#endif

#ifdef FACTOR_LEVEL
    std::vector<BCAST<T>> bcastlist_new;
    commandlist.push_back(Command<T>(comm_temp, command::run));
#endif

    //  EXIT CONDITION
    if(level == numlevel) {
      if(printid == ROOT)
         printf("************************************ leaf level %d groupsize %d\n", level, groupsize[level - 1]);
      for(auto bcast : bcastlist) {
        for(auto recvid : bcast.recvid) {
          int sendid = bcast.sendid;
          comm_temp->add(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, sendid, recvid);
        }
      }
      if(printid == ROOT)
        printf("\n");
#ifdef FACTOR_LOCAL
      if(level <= nodelevel)
        commandlist.push_back(Command<T>(comm_temp, command::start));
      waitlist.push_back(Command<T>(comm_temp, command::wait));
#endif
      return;
    }

    int numgroup = numproc / groupsize[level];

    // LOCAL COMMUNICATIONS
    {
#ifdef FACTOR_LOCAL
      std::vector<BCAST<T>> bcastlist_new;
#endif
      for(auto bcast : bcastlist) {
        int sendgroup = bcast.sendid / groupsize[level];
        for(int recvgroup = 0; recvgroup < numgroup; recvgroup++) {
          if(sendgroup == recvgroup) {
            std::vector<int> recvids;
            for(auto recvid : bcast.recvid) {
              int group = recvid / groupsize[level];
              if(group == recvgroup)
                recvids.push_back(recvid);
            }
            if(recvids.size()) {
              // if(printid == ROOT)
              //  printf("level %d groupsize %d numgroup %d sendgroup %d recvgroup %d recvid %d\n", level, groupsize[level], numgroup, sendgroup, recvgroup, bcast.sendid);
              bcastlist_new.push_back(BCAST<T>(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, bcast.sendid, recvids));
            }
          }
        }
      }
#ifdef FACTOR_LOCAL
      bcast_tree(comm_mpi, numlevel, groupsize, lib, bcastlist_new, commlist, level + 1, commandlist, waitlist, nodelevel);
#endif
    }

    // GLOBAL COMMUNICATIONS
    {
#ifdef FACTOR_LOCAL
      std::vector<BCAST<T>> bcastlist_new;
#endif
      for(int recvgroup = 0; recvgroup < numgroup; recvgroup++) {
        for(auto bcast : bcastlist) {
          int sendgroup = bcast.sendid / groupsize[level];
          if(sendgroup != recvgroup) {
            std::vector<int> recvids;
            for(auto recvid : bcast.recvid) {
              if(recvid / groupsize[level] == recvgroup)
                recvids.push_back(recvid);
            }
            if(recvids.size()) {
              int recvid = recvgroup * groupsize[level] + bcast.sendid % groupsize[level];
              // if(printid == ROOT)
              //  printf("level %d groupsize %d numgroup %d sendgroup %d recvgroup %d recvid %d\n", level, groupsize[level], numgroup, sendgroup, recvgroup, recvid);
              T *recvbuf;
              int recvoffset;
              bool found = false;
              for(auto it = recvids.begin(); it != recvids.end(); ++it) {
                if(*it == recvid) {
                  if(printid == ROOT)
                    printf("******************************************************************************************* found recvid %d\n", *it);
                  recvbuf = bcast.recvbuf;
                  recvoffset = bcast.recvoffset;
                  found = true;
                  recvids.erase(it);
                  break;
                }
              }
              if(!found && myid == recvid) {
#ifdef PORT_CUDA
                cudaMalloc(&recvbuf, bcast.count * sizeof(T));
#elif defined PORT_HIP
                hipMalloc(&recvbuf, bcast.count * sizeof(T));
#endif
                recvoffset = 0;
                printf("^^^^^^^^^^^^^^^^^^^^^^^ recvid %d myid %d allocates recvbuf %p equal %d\n", recvid, myid, recvbuf, myid == recvid);
              }
              comm_temp->add(bcast.sendbuf, bcast.sendoffset, recvbuf,  recvoffset, bcast.count, bcast.sendid, recvid);
              if(recvids.size()) {
                bcastlist_new.push_back(BCAST<T>(recvbuf, recvoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, recvid, recvids));
              }
            }
          }
        }
      }
#ifdef FACTOR_LOCAL
      if(level <= nodelevel)
        commandlist.push_back(Command<T>(comm_temp, command::start));
      commandlist.push_back(Command<T>(comm_temp, command::wait));
      bcast_tree(comm_mpi, numlevel, groupsize, lib, bcastlist_new, commlist, level + 1, commandlist, waitlist, nodelevel);
#endif
    }
#ifdef FACTOR_LEVEL
    bcast_tree(comm_mpi, numlevel, groupsize, lib, bcastlist_new, commlist, level + 1, commandlist, waitlist, nodelevel);
#endif
  }

  /*template <typename T>
  void striped(const MPI_Comm &comm_mpi, int numlevel, int groupsize[], CommBench::library lib[], const std::vector<BCAST<T>> &bcastlist, std::list<CommBench::Comm<T>*> &commlist, std::list<Command<T>> &commandlist) {

    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    // SEPARATE INTRA AND INTER NODES
    std::vector<BCAST<T>> bacstlist_intra;
    std::vector<BCAST<T>> bcastlist_inter;
    for(auto &p2p : bcastlist) {
      int sendid = p2p.sendid;
      for(auto recvid = p2p.recvid) {
      }
      if(p2p.sendid / groupsize[0] == p2p.recvid / groupsize[0])
        addlist_intra.push_back(p2p);
      else
        addlist_inter.push_back(p2p);
    }
    if(printid == ROOT) {
      printf("broadcast striping group size: %d numgroups: %d\n", groupsize[0], numproc / groupsize[0]);
      printf("number of intra-comm: %zu inter-comm: %zu\n", addlist_intra.size(), addlist_inter.size());
    }


  }*/

 #define OVERLAP
  template <typename T>
  void striped(const MPI_Comm &comm_mpi, int numlevel, int groupsize[], CommBench::library lib[], const std::vector<P2P<T>> &addlist, std::list<CommBench::Comm<T>*> &commlist, std::list<Command<T>> &commandlist) {

    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    // SEPARATE INTRA AND INTER NODES
    std::vector<P2P<T>> addlist_intra;
    std::vector<P2P<T>> addlist_inter;
    for(auto &p2p : addlist) {
      if(p2p.sendid / groupsize[0] == p2p.recvid / groupsize[0])
        addlist_intra.push_back(p2p);
      else
        addlist_inter.push_back(p2p);
    }
    if(printid == ROOT) {
      printf("point-to-point striping group size: %d numgroups: %d\n", groupsize[0], numproc / groupsize[0]);
      printf("number of intra-comm: %zu inter-comm: %zu\n", addlist_intra.size(), addlist_inter.size());
    }

    std::vector<CommBench::Comm<T>*> split;
    std::vector<CommBench::Comm<T>*> merge;
    std::vector<CommBench::Comm<T>*> inter;
    std::vector<CommBench::Comm<T>*> intra;

    // INTER-NODE MIXED STRIPING
    if(addlist_inter.size())
    {
      inter.push_back(new CommBench::Comm<T>(comm_mpi, lib[0]));
      for(int level = 1; level < numlevel; level++) {
        split.push_back(new CommBench::Comm<T>(comm_mpi, lib[level]));
        merge.push_back(new CommBench::Comm<T>(comm_mpi, lib[level]));
      }
      for(auto &p2p : addlist_inter) {
        int sendgroup = p2p.sendid / groupsize[0];
        int recvgroup = p2p.recvid / groupsize[0];
        int mygroup = myid / groupsize[0];
        T *sendbuf_temp;
        T *recvbuf_temp;
        size_t splitcount = p2p.count / groupsize[0];
#ifdef PORT_CUDA
        if(mygroup == sendgroup && myid != p2p.sendid)
          cudaMalloc(&sendbuf_temp, splitcount * sizeof(T));
        if(mygroup == recvgroup && myid != p2p.recvid)
          cudaMalloc(&recvbuf_temp, splitcount * sizeof(T));
#elif defined PORT_HIP
        if(mygroup == sendgroup && myid != p2p.sendid) 
          hipMalloc(&sendbuf_temp, splitcount * sizeof(T));
        if(mygroup == recvgroup && myid != p2p.recvid) 
          hipMalloc(&recvbuf_temp, splitcount * sizeof(T));
#endif
        // SPLIT
        for(int p = 0; p < groupsize[0]; p++) {
          int recver = sendgroup * groupsize[0] + p;
          if(printid == ROOT)
            printf("split ");
          if(recver != p2p.sendid) {
            for(int level = numlevel - 1; level > -1; level--)
              if(recver / groupsize[level] == p2p.sendid / groupsize[level]) {
                if(printid == ROOT)
                  printf("level %d ", level);
                split[level - 1]->add(p2p.sendbuf, p2p.sendoffset + p * splitcount, sendbuf_temp, 0, splitcount, p2p.sendid, recver);
                break;
              }
          }
          else
            if(printid == ROOT)
              printf("* skip self\n");
        }
        // INTER
        for(int p = 0; p < groupsize[0]; p++) {
          if(printid == ROOT)
            printf("inter ");
          int sender = sendgroup * groupsize[0] + p;
          int recver = recvgroup * groupsize[0] + p;
          if(sender == p2p.sendid && recver == p2p.recvid)
            inter[0]->add(p2p.sendbuf, p2p.sendoffset + p * splitcount, p2p.recvbuf, p2p.recvoffset + p * splitcount, splitcount, sender, recver);
          if(sender != p2p.sendid && recver == p2p.recvid)
            inter[0]->add(sendbuf_temp, 0, p2p.recvbuf, p2p.recvoffset + p * splitcount, splitcount, sender, recver);
          if(sender == p2p.sendid && recver != p2p.recvid)
            inter[0]->add(p2p.sendbuf, p2p.sendoffset + p * splitcount, recvbuf_temp, 0, splitcount, sender, recver);
          if(sender != p2p.sendid && recver != p2p.recvid)
            inter[0]->add(sendbuf_temp, 0, recvbuf_temp, 0, splitcount, sender, recver);
        }
        // MERGE
        for(int p = 0; p < groupsize[0]; p++) {
          int sender = recvgroup * groupsize[0] + p;
          if(printid == ROOT)
            printf("merge ");
          if(sender != p2p.recvid) {
            for(int level = numlevel - 1; level > -1; level--)
              if(sender / groupsize[level] == p2p.recvid / groupsize[level]) {
                if(printid == ROOT)
                  printf("level %d ", level);
                merge[level - 1]->add(recvbuf_temp, 0, p2p.recvbuf, p2p.recvoffset + p * splitcount, splitcount, sender, p2p.recvid);
                break;
              }
          }
          else
            if(printid == ROOT)
              printf(" * skip self\n");
        }
      }
    }
    // INTRA-NODE MIXED FLAT
    if(addlist_intra.size()) {
      for(int level = 0; level < numlevel; level++)
        intra.push_back(new CommBench::Comm<T>(comm_mpi, lib[level]));
      for(auto &p2p : addlist_intra) {
        for(int level = numlevel - 1; level > -1; level--) {
          if(p2p.sendid / groupsize[level] == p2p.recvid / groupsize[level]) {
            if(printid == ROOT)
              printf("level %d ", level);
            intra[level]->add(p2p.sendbuf, p2p.sendoffset, p2p.recvbuf, p2p.recvoffset, p2p.count, p2p.sendid, p2p.recvid);
            break;
          }
        }
      }

      if(printid == ROOT)
        printf("\n");
    }

    if(printid == ROOT) {
      printf("split size: %zu\n", split.size());
      printf("inter size: %zu\n", inter.size());
      printf("merge size: %zu\n", merge.size());
      printf("intra size: %zu\n", intra.size());
    }
    // PUSH SPLIT
    for(auto comm : split)
      commlist.push_back(comm);
    // PUSH INTER
    for(auto comm : inter)
      commlist.push_back(comm);
    // PUSH MERGE
    for(auto comm : merge)
      commlist.push_back(comm);
    // PUSH INTRA
    for(auto comm : intra)
      commlist.push_back(comm);

    // START SPLIT
    for(auto comm : split)
      commandlist.push_back(Command<T>(comm, command::start));
    // WAIT FOR SPLIT
    for(auto comm : split)
      commandlist.push_back(Command<T>(comm, command::wait));
    // START INTER-COMMUNICATION
    for(auto comm : inter)
      commandlist.push_back(Command<T>(comm, command::start));
#ifdef OVERLAP
    // START INTRA-COMMUNICATION
    for(auto comm : intra)
      commandlist.push_back(Command<T>(comm, command::start));
#endif
    // WAIT FOR INTER-COMMUNICATION
    for(auto comm : inter)
      commandlist.push_back(Command<T>(comm, command::wait));
    // START MERGE
    for(auto comm : merge)
      commandlist.push_back(Command<T>(comm, command::start));
#ifdef OVERLAP
    // WAIT FOR INTRA-COMMUNICATION
    for(auto comm : intra)
      commandlist.push_back(Command<T>(comm, command::wait));
#endif
    // WAIT FOR MERGE
    for(auto comm : merge)
      commandlist.push_back(Command<T>(comm, command::wait));
#ifndef OVERLAP
    // START INTRA-COMMUNICATION
    for(auto comm : intra)
      commandlist.push_back(Command<T>(comm, command::start));
    // WAIT FOR INTRA-COMMUNICATION
    for(auto comm : intra)
      commandlist.push_back(Command<T>(comm, command::wait));
#endif
  }

  template <typename T>
  class Comm {

    const MPI_Comm comm_mpi;

    std::vector<P2P<T>> addlist;
    std::vector<BCAST<T>> bcastlist;

    std::list<CommBench::Comm<T>*> commlist;
    std::list<Command<T>> commandlist;
    std::list<T*> bufferlist;

    // PIPELINING
    std::vector<std::list<CommBench::Comm<T>*>> comm_batch;

    public:

    Comm(const MPI_Comm &comm_mpi) : comm_mpi(comm_mpi) {}

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
      // INIT POINT-TO-POINT
      {
        // PARTITION INTO BATCHES
        std::vector<std::vector<P2P<T>>> p2p_batch(numbatch);
        for(auto &p2p : addlist) {
          int batchsize = p2p.count / numbatch;
          for(int batch = 0; batch < numbatch; batch++)
            p2p_batch[batch].push_back(P2P<T>(p2p.sendbuf, p2p.sendoffset + batch * batchsize, p2p.recvbuf, p2p.recvoffset + batch * batchsize, batchsize, p2p.sendid, p2p.recvid));
        }
        std::vector<std::list<CommBench::Comm<T>*>> comm_batch(numbatch);
        // ADD INITIAL DUMMY COMMUNICATORS INTO THE PIPELINE
        for(int batch = 0; batch < numbatch; batch++)
          for(int c = 0; c < batch; c++)
            comm_batch[batch].push_back(new CommBench::Comm<T>(comm_mpi, CommBench::MPI));
	// ADD DUTY COMMUNICATORS INTO THE PIPELINE
        for(int batch = 0; batch < numbatch; batch++) {
          std::list<Command<T>> commandlist;
          striped(comm_mpi, numlevel, groupsize, lib, p2p_batch[batch], comm_batch[batch], commandlist);
        }
        this->comm_batch = comm_batch;
        // striped(comm_mpi, numlevel, groupsize, lib, addlist, commlist, commandlist);
      }
      // INIT BROADCAST
      {
        // PARTITION INTO BATCHES
        std::vector<std::vector<BCAST<T>>> bcast_batch(numbatch);
        for(auto &bcast : bcastlist) {
          int batchsize = bcast.count / numbatch;
          for(int batch = 0; batch < numbatch; batch++)
            bcast_batch[batch].push_back(BCAST<T>(bcast.sendbuf, bcast.sendoffset + batch * batchsize, bcast.recvbuf, bcast.recvoffset + batch * batchsize, batchsize, bcast.sendid, bcast.recvid));
        }
        std::vector<std::list<CommBench::Comm<T>*>> comm_batch(numbatch);
        // ADD INITIAL DUMMY COMMUNICATORS INTO THE PIPELINE
        for(int batch = 0; batch < numbatch; batch++)
          for(int c = 0; c < batch; c++)
            comm_batch[batch].push_back(new CommBench::Comm<T>(comm_mpi, CommBench::MPI));
	// ADD DUTY COMMUNICATOPNS INTO THE PIPELINE
	for(int batch = 0; batch < numbatch; batch++) {
          std::list<Command<T>> waitlist;
	  std::list<Command<T>> commandlist;
          ExaComm::bcast_tree(comm_mpi, numlevel, groupsize, lib, bcast_batch[batch], comm_batch[batch], 1, commandlist, waitlist, 1);
        }
        this->comm_batch = comm_batch;
      }
    };

    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
      addlist.push_back(P2P<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvid));
    }

    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, std::vector<int> &recvid) {
      bcastlist.push_back(BCAST<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvid));
    }

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

template <typename T>
void measure(size_t count, int warmup, int numiter, Comm<T> &comm) {

  int myid;
  int numproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  int numthread = -1;
  #pragma omp parallel
  #pragma omp master
  numthread = omp_get_num_threads();

  double times[numiter];
  if(myid == ROOT)
    printf("%d warmup iterations (in order) numthread %d:\n", warmup, numthread);
  for (int iter = -warmup; iter < numiter; iter++) {

#ifdef PORT_CUDA
    cudaDeviceSynchronize();
#elif defined PORT_HIP
    hipDeviceSynchronize();
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();
   //  comm.run_commandlist();
    // comm.run_batch();
    comm.overlap_batch();
    time = MPI_Wtime() - time;

    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(iter < 0) {
      if(myid == ROOT)
        printf("warmup: %e\n", time);
    }
    else
      times[iter] = time;
  }
  std::sort(times, times + numiter,  [](const double & a, const double & b) -> bool {return a < b;});

  if(myid == ROOT) {
    printf("%d measurement iterations (sorted):\n", numiter);
    for(int iter = 0; iter < numiter; iter++) {
      printf("time: %.4e", times[iter]);
      if(iter == 0)
        printf(" -> min\n");
      else if(iter == numiter / 2)
        printf(" -> median\n");
      else if(iter == numiter - 1)
        printf(" -> max\n");
      else
        printf("\n");
    }
    printf("\n");
    double minTime = times[0];
    double medTime = times[numiter / 2];
    double maxTime = times[numiter - 1];
    double avgTime = 0;
    for(int iter = 0; iter < numiter; iter++)
      avgTime += times[iter];
    avgTime /= numiter;
    double data = count * sizeof(int);
    if (data < 1e3)
      printf("data: %d bytes\n", (int)data);
    else if (data < 1e6)
      printf("data: %.4f KB\n", data / 1e3);
    else if (data < 1e9)
      printf("data: %.4f MB\n", data / 1e6);
    else if (data < 1e12)
      printf("data: %.4f GB\n", data / 1e9);
    else
      printf("data: %.4f TB\n", data / 1e12);
    printf("minTime: %.4e us, %.4e s/GB, %.4e GB/s\n", minTime * 1e6, minTime / data * 1e9, data / minTime / 1e9);
    printf("medTime: %.4e us, %.4e s/GB, %.4e GB/s\n", medTime * 1e6, medTime / data * 1e9, data / medTime / 1e9);
    printf("maxTime: %.4e us, %.4e s/GB, %.4e GB/s\n", maxTime * 1e6, maxTime / data * 1e9, data / maxTime / 1e9);
    printf("avgTime: %.4e us, %.4e s/GB, %.4e GB/s\n", avgTime * 1e6, avgTime / data * 1e9, data / avgTime / 1e9);
    printf("\n");
  }
}

template <typename T>
void validate(int *sendbuf_d, int *recvbuf_d, size_t count, int pattern, Comm<T> &comm) {

  int myid;
  int numproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  int *recvbuf;
  int *sendbuf;
#ifdef PORT_CUDA
  cudaMallocHost(&sendbuf, count * numproc * sizeof(int));
  cudaMallocHost(&recvbuf, count * numproc * sizeof(int));
#elif defined PORT_HIP
  hipHostMalloc(&sendbuf, count * numproc * sizeof(int));
  hipHostMalloc(&recvbuf, count * numproc * sizeof(int));
#endif
  
  for(int p = 0; p < numproc; p++)
    for(size_t i = p * count; i < (p + 1) * count; i++)
      sendbuf[i] = i;
#ifdef PORT_CUDA
  cudaMemcpy(sendbuf_d, sendbuf, count * sizeof(int) * numproc, cudaMemcpyHostToDevice);
  cudaMemset(recvbuf_d, -1, count * numproc * sizeof(int));
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaDeviceSynchronize();
#elif defined PORT_HIP
  hipMemcpy(sendbuf_d, sendbuf, count * sizeof(int) * numproc, hipMemcpyHostToDevice);
  hipMemset(recvbuf_d, -1, count * numproc * sizeof(int));
  hipStream_t stream;
  hipStreamCreate(&stream);
  hipDeviceSynchronize();
#endif
  memset(recvbuf, -1, count * numproc * sizeof(int));

  //comm.run_commandlist();
  //comm.run_commbatch();
  // comm.run_batch();
  comm.overlap_batch();

#ifdef PORT_CUDA
  cudaMemcpyAsync(recvbuf, recvbuf_d, count * sizeof(int) * numproc, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
#elif defined PORT_HIP
  hipMemcpyAsync(recvbuf, recvbuf_d, count * sizeof(int) * numproc, hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);
#endif

  bool pass = true;
  switch(pattern) {
    case 0:
      {
        if(myid == 0) printf("VERIFY P2P\n");
        if(myid == 15) {
          for(size_t i = 0; i < count; i++) {
            // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
            if(recvbuf[i] != i)
              pass = false;
          }
        }
      }
      break;
    case 1:
      {
        if(myid == ROOT) printf("VERIFY GATHER\n");
        if(myid == ROOT) {
          for(int p = 0; p < numproc; p++)
            for(size_t i = 0; i < count; i++) {
              // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
              if(recvbuf[p * count + i] != i)
                pass = false;
            }
        }
      }
      break;
    case 2:
      {
        if(myid == ROOT) printf("VERIFY SCATTER ROOT = %d\n", ROOT);
        for(size_t i = 0; i < count; i++) {
          // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
          if(recvbuf[i] != myid * count + i)
            pass = false;
        }
      }
      break;
    case 4:
      {
        if(myid == ROOT) printf("VERIFY BCAST ROOT = %d\n", ROOT);
        for(size_t i = 0; i < count; i++) {
          // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
          if(recvbuf[i] != i)
            pass = false;
        }
      }
      break;
    case 5:
      {
        if(myid == ROOT) printf("VERIFY ALL-TO-ALL\n");
        for(int p = 0; p < numproc; p++)
          for(size_t i = 0; i < count; i++) {
            // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
            if(recvbuf[p * count + i] != myid * count + i)
              pass = false;
          }
      }
      break;
    case 7:
      {
        if(myid == ROOT) printf("VERIFY ALL-GATHER\n");
        for(int p = 0; p < numproc; p++)
          for(size_t i = 0; i < count; i++) {
            // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
            if(recvbuf[p * count + i] != i)
              pass = false;
          }
      }
      break;
  }

  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
  if(myid == ROOT) {
    if(pass) 
      printf("PASSED!\n");
    else 
      printf("FAILED!!!\n");
  }

#ifdef PORT_CUDA
  cudaFreeHost(sendbuf);
  cudaFreeHost(recvbuf);
#elif defined PORT_HIP
  hipHostFree(sendbuf);
  hipHostFree(recvbuf);
#endif
};

}
