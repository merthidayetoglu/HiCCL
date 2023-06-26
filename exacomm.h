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

  template <typename T>
  struct Command {
    public:
    CommBench::Comm<T> *comm;
    command com;
    Command(CommBench::Comm<T> *comm, command com) : comm(comm), com(com) {}
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
  void run_command(std::list<Command<T>> &commandlist) {
    for(auto comm : commandlist) {
      switch(comm.com) {
        case(command::start) :
          comm.comm->start();
          break;
        case(command::wait) :
          comm.comm->wait();
          break;
        case(command::run) :
          comm.comm->run();
          break;
      }
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

    BCAST(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int numrecv, int recvid[])
    : sendbuf(sendbuf), sendoffset(sendoffset), recvbuf(recvbuf), recvoffset(recvoffset), count(count), sendid(sendid) {
      for(int i = 0; i < numrecv; i++)
        this->recvid.push_back(recvid[i]);
    }
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

    CommBench::Comm<T> *comm_temp = new CommBench::Comm<T>(comm_mpi, lib[level-1]);
    commlist.push_back(comm_temp);

#ifdef FACTOR_LOCAL
    if(level > nodelevel)
      commandlist.push_back(Command<T>(comm_temp, command::start));
#endif

#ifdef FACTOR_LEVEL
    std::vector<BCAST<T>> bcastlist_new;
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
              if(printid == ROOT)
                printf("level %d groupsize %d numgroup %d sendgroup %d recvgroup %d recvid %d\n", level, groupsize[level], numgroup, sendgroup, recvgroup, bcast.sendid);
              bcastlist_new.push_back(BCAST<T>(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, bcast.sendid, recvids.size(), recvids.data()));
            }
          }
        }
      }
#ifdef FACTOR_LOCAL
      if(bcastlist_new.size()) {
        bcast_tree(comm_mpi, numlevel, groupsize, lib, bcastlist_new, commlist, level + 1, commandlist, waitlist, nodelevel);
      }
#endif
    }

#ifdef FACTOR_LOCAL
    if(level <= nodelevel)
      commandlist.push_back(Command<T>(comm_temp, command::start));
    commandlist.push_back(Command<T>(comm_temp, command::wait));
#endif

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
              if(printid == ROOT)
                printf("level %d groupsize %d numgroup %d sendgroup %d recvgroup %d recvid %d\n", level, groupsize[level], numgroup, sendgroup, recvgroup, recvid);
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
                bcastlist_new.push_back(BCAST<T>(recvbuf, recvoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, recvid, recvids.size(), recvids.data()));
              }
            }
          }
        }
      }
#ifdef FACTOR_LOCAL
      if(bcastlist_new.size())
        bcast_tree(comm_mpi, numlevel, groupsize, lib, bcastlist_new, commlist, level + 1, commandlist, waitlist, nodelevel);
#endif
    }

#ifdef FACTOR_LEVEL
    if(bcastlist_new.size())
      bcast_tree(comm_mpi, numlevel, groupsize, lib, bcastlist_new, commlist, level + 1, commandlist, waitlist, nodelevel);
#endif
  }

  template <typename T>
  class Comm {

    const MPI_Comm comm_mpi;

    std::vector<P2P<T>> addlist;
    std::vector<BCAST<T>> bcastlist;

    std::vector<CommBench::Comm<T>> comm_inter;
    std::vector<CommBench::Comm<T>> comm_intra;
    std::vector<CommBench::Comm<T>> comm_split;
    std::vector<CommBench::Comm<T>> comm_merge;

    std::vector<T*> sendbuf_inter;
    std::vector<T*> recvbuf_inter;

    void start() {
      for(auto &comm : comm_split)
        comm.run();
      for(auto &comm : comm_inter)
        comm.launch();
      for(auto &comm : comm_intra)
        comm.launch();
    }
    void wait() {
      for(auto &comm : comm_intra)
        comm.wait();
      for(auto &comm : comm_inter)
        comm.wait();
      for(auto &comm : comm_merge)
        comm.run();
    }

    void init_striped(int numlevel, int groupsize[], CommBench::library lib[]) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);

      for(int level = 0; level < numlevel; level++)
        comm_intra.push_back(CommBench::Comm<T>(comm_mpi, lib[level]));

      std::vector<P2P<T>> addlist_inter;

      for(auto &p2p : addlist) {
        bool found = false;
        for(int level = numlevel - 1; level > -1; level--)
          if(p2p.sendid / groupsize[level] == p2p.recvid / groupsize[level]) {
            if(myid == ROOT)
              printf("level %d ", level + 1);
            comm_intra[level].add(p2p.sendbuf, p2p.sendoffset, p2p.recvbuf, p2p.recvoffset, p2p.count, p2p.sendid, p2p.recvid);
            found = true;
            break;
          }
        if(!found) {
          if(myid == ROOT)
            printf("level 0  *  (%d -> %d) sendoffset %lu recvoffset %lu count %lu \n", p2p.sendid, p2p.recvid, p2p.sendoffset, p2p.recvoffset, p2p.count);
          addlist_inter.push_back(P2P<T>(p2p.sendbuf, p2p.sendoffset, p2p.recvbuf, p2p.recvoffset, p2p.count, p2p.sendid, p2p.recvid));
        }
      }
      if(myid == ROOT) {
        printf("* to be splitted\n");
	printf("\n");
      }

      comm_split.push_back(CommBench::Comm<T>(comm_mpi, lib[0]));
      comm_merge.push_back(CommBench::Comm<T>(comm_mpi, lib[0]));

      for(auto &p2p : addlist_inter) {
        int sendgroup = p2p.sendid / groupsize[0];
        int recvgroup = p2p.recvid / groupsize[0];
        int mygroup = myid / groupsize[0];
        T *sendbuf_temp;
        T *recvbuf_temp;
        size_t splitcount = p2p.count / groupsize[0];
#ifdef PORT_CUDA
        if(mygroup == sendgroup && myid != p2p.sendid) {
          cudaMalloc(&sendbuf_temp, splitcount * sizeof(T));
	  sendbuf_inter.push_back(sendbuf_temp);
        }
	if(mygroup == recvgroup && myid != p2p.recvid) {
          cudaMalloc(&recvbuf_temp, splitcount * sizeof(T));
	  recvbuf_inter.push_back(recvbuf_temp);
        }
#elif defined PORT_HIP
#endif
        for(int p = 0; p < groupsize[0]; p++) {
          int recver = sendgroup * groupsize[0] + p;
          if(myid == ROOT)
            printf("split ");
          if(recver != p2p.sendid)
            comm_split[0].add(p2p.sendbuf, p2p.sendoffset + p * splitcount, sendbuf_temp, 0, splitcount, p2p.sendid, recver);
	  else
            if(myid == ROOT)
              printf(" * \n");
        }
	for(int p = 0; p < groupsize[0]; p++) {
          if(myid == ROOT)
            printf("inter ");
          int sender = sendgroup * groupsize[0] + p;
          int recver = recvgroup * groupsize[0] + p;
	  if(sender == p2p.sendid && recver == p2p.recvid)
            comm_inter[0].add(p2p.sendbuf, p2p.sendoffset + p * splitcount, p2p.recvbuf, p2p.recvoffset + p * splitcount, splitcount, sender, recver);
	  if(sender != p2p.sendid && recver == p2p.recvid)
            comm_inter[0].add(sendbuf_temp, 0, p2p.recvbuf, p2p.recvoffset + p * splitcount, splitcount, sender, recver);
	  if(sender == p2p.sendid && recver != p2p.recvid)
            comm_inter[0].add(p2p.sendbuf, p2p.sendoffset + p * splitcount, recvbuf_temp, 0, splitcount, sender, recver);
	  if(sender != p2p.sendid && recver != p2p.recvid)
            comm_inter[0].add(sendbuf_temp, 0, recvbuf_temp, 0, splitcount, sender, recver);
        }
        for(int p = 0; p < groupsize[0]; p++) {
          int sender = recvgroup * groupsize[0] + p;
          if(myid == ROOT)
            printf("merge ");
          if(sender != p2p.recvid)
            comm_merge[0].add(recvbuf_temp, 0, p2p.recvbuf, p2p.recvoffset + p * splitcount, splitcount, sender, p2p.recvid);
	  else
            if(myid == ROOT)
              printf(" * \n");
        }
      }
      if(myid == ROOT) {
        printf("* pruned\n");
        printf("\n");
      }
    }

    void init_mixed(int numlevel, int groupsize[], CommBench::library lib[]) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);

      for(int level = 0; level < numlevel; level++)
        comm_intra.push_back(CommBench::Comm<T>(comm_mpi, lib[level]));

      for(auto &p2p : addlist) {
        bool found = false;
        for(int level = numlevel - 1; level > -1; level--)
          if(p2p.sendid / groupsize[level] == p2p.recvid / groupsize[level]) {
            if(myid == ROOT)
              printf("level %d ", level + 1);
            comm_intra[level].add(p2p.sendbuf, p2p.sendoffset, p2p.recvbuf, p2p.recvoffset, p2p.count, p2p.sendid, p2p.recvid);
	    found = true;
            break;
          }
        if(!found) {
          if(myid == ROOT)
            printf("level 0 ");
          comm_inter[0].add(p2p.sendbuf, p2p.sendoffset, p2p.recvbuf, p2p.recvoffset, p2p.count, p2p.sendid, p2p.recvid);
        }
      }
      if(myid == ROOT)
        printf("\n");
    }

    public:

    Comm(const MPI_Comm &comm_mpi, CommBench::library lib)
    : comm_mpi(comm_mpi) {
      comm_inter.push_back(CommBench::Comm<T>(comm_mpi, lib));
    }

    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
      addlist.push_back(P2P<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvid));
    }

    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int numrecv, int recvid[]) {
      bcastlist.push_back(BCAST<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, numrecv, recvid));
    }

    void init_hierarchical(int numlevel, int groupsize[], CommBench::library lib[], int stripelevel) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);

      // P2P COMMUNICATIONS
      for(int level = 0; level < numlevel; level++)
        comm_intra.push_back(CommBench::Comm<T>(comm_mpi, lib[level]));

      for(auto &p2p : addlist) {
        bool found = false;
        for(int level = numlevel - 1; level > stripelevel - 1; level--)
          if(p2p.sendid / groupsize[level] == p2p.recvid / groupsize[level]) {
            if(myid == ROOT)
              printf("level %d groupsize %d", level + 1, groupsize[level]);
            comm_intra[level].add(p2p.sendbuf, p2p.sendoffset, p2p.recvbuf, p2p.recvoffset, p2p.count, p2p.sendid, p2p.recvid);
            found = true;
            break;
          }
        if(!found) {
          if(myid == ROOT)
            printf("level 0 ");
          comm_inter[0].add(p2p.sendbuf, p2p.sendoffset, p2p.recvbuf, p2p.recvoffset, p2p.count, p2p.sendid, p2p.recvid);
        }
      }
      if(myid == ROOT)
        printf("\n");

      // BCAST COMMUNICATIONS
      for(auto &bcast : bcastlist) {
        int recvid_inter[bcast.recvid.size()];
        int sendid = bcast.sendid;
        for(auto &recvid : bcast.recvid) {
          for(int level = 0; level < numlevel; level++) {
            int numgroups = numproc / groupsize[0];
            printf("sendid %d recvid %d level %d groupsize %d numgroups %d\n", sendid, recvid, level, groupsize[level], numgroups);
          }
        }
      }
    }

    void init_flat() {
      init_mixed(0, NULL, NULL);
    }

    void init_mixed(int groupsize, CommBench::library lib) {
      init_mixed(1, &groupsize, &lib);
    };

    void init_bcast(int groupsize, CommBench::library lib) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);

      comm_intra.push_back(CommBench::Comm<T>(comm_mpi, lib));

      std::vector<BCAST<T>> bcast_inter;

      for(auto &bcast : bcastlist) {
        int numrecv_inter = 0;
        int recvid_inter[bcast.recvid.size()];
        int sendid = bcast.sendid;
        for(auto &recvid : bcast.recvid) {
          if(sendid / groupsize == recvid / groupsize) {
            if(myid == ROOT)
              printf("level 1 ");
            comm_intra[0].add(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, sendid, recvid);
          }
          else {
            if(myid == ROOT)
              printf("level 0  *  (%d -> %d) sendoffset %lu recvoffset %lu count %lu \n", sendid, recvid, bcast.sendoffset, bcast.recvoffset, bcast.count);
            recvid_inter[numrecv_inter] = recvid;
            numrecv_inter++;
          }
        }
        bcast_inter.push_back(BCAST<T>(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, sendid, numrecv_inter, recvid_inter));
      }
      if(myid == ROOT) {
        printf("* to be splitted and broadcasted\n");
        printf("\n");
      }

      comm_split.push_back(CommBench::Comm<T>(comm_mpi, lib));
      comm_merge.push_back(CommBench::Comm<T>(comm_mpi, lib));

      int mygroup = myid / groupsize;
      int numgroup = numproc / groupsize;

      for(auto &bcast : bcast_inter) {
        int sendid = bcast.sendid;
        int sendgroup = bcast.sendid / groupsize;
        size_t splitcount = bcast.count / groupsize;
        T *sendbuf_temp;
        T *recvbuf_temp;
        // SENDING GROUP
        if(bcast.recvid.size()) {
          if(mygroup == sendgroup) {
#ifdef PORT_CUDA
            cudaMalloc(&sendbuf_temp, splitcount * sizeof(T));
#elif defined PORT_HIP
            hipMalloc(&sendbuf_temp, splitcount * sizeof(T));
#endif
            sendbuf_inter.push_back(sendbuf_temp);
          }
	  for(int p = 0; p < groupsize; p++) {
            if(myid == ROOT)
              printf("split ");
            int recvid = sendgroup * groupsize + p;
            comm_split[0].add(bcast.sendbuf, bcast.sendoffset + p * splitcount, sendbuf_temp, 0, splitcount, sendid, recvid);
          }
        }
	// AUXILIARY DATA STRUCTURES
        int numrecv_group[numgroup];
        int recvproc_group[numgroup][bcast.recvid.size()];
        memset(numrecv_group, 0, numgroup * sizeof(int));
        for(auto &recvid : bcast.recvid) {
          int recvgroup = recvid / groupsize;
          recvproc_group[recvgroup][numrecv_group[recvgroup]] = recvid;
          numrecv_group[recvgroup]++;
        }
	// RECEIVING GROUPS
        for(int g = 0; g < numgroup; g++) {
          if(numrecv_group[g]) {
            if(g == mygroup) {
#ifdef PORT_CUDA
              cudaMalloc(&recvbuf_temp, splitcount * sizeof(T));
#elif defined PORT_HIP
              hipMalloc(&recvbuf_temp, splitcount * sizeof(T));
#endif
              recvbuf_inter.push_back(recvbuf_temp);
	    }
            for(int p = 0; p < groupsize; p++) {
              if(myid == ROOT)
                printf("inter ");
              int recvid = g * groupsize + p;
              int sendid = sendgroup * groupsize + p;
              comm_inter[0].add(sendbuf_temp, 0, recvbuf_temp, 0, splitcount, sendid, recvid);
            }
            for(int pp = 0; pp < numrecv_group[g]; pp++) {
              if(myid == ROOT)
                printf("merge ");
              int recvid = recvproc_group[g][pp];
              for(int p = 0; p < groupsize; p++) {
                int sendid = g * groupsize + p;
                comm_merge[0].add(recvbuf_temp, 0, bcast.recvbuf, bcast.recvoffset + p * splitcount, splitcount, sendid, recvid);
	      }   
            }
          }
        }
      }
    }
    void run() {
      start();
      wait();
    }
    void measure(int numwarmup, int numiter) {
      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);

      for(auto &comm : comm_split) {
        if(myid == ROOT)
          printf("******************** measure split map ");
        comm.measure(numwarmup, numiter);
      }
      for(auto &comm : comm_inter) {
        if(myid == ROOT)
          printf("******************** measure inter-group ");
        comm.measure(numwarmup, numiter);
      }
      for(auto &comm : comm_merge) {
        if(myid == ROOT)
          printf("******************** measure merge map ");
        comm.measure(numwarmup, numiter);
      }
      for(auto &comm : comm_intra) {
        if(myid == ROOT)
          printf("******************** measure intra-group ");
        comm.measure(numwarmup, numiter);
      }
    }
  };
}

