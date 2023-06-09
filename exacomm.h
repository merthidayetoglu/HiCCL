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

#include <vector>


namespace ExaComm {

  int printid;

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
      for(int p = 0; p < numrecv; p++)
        this->recvid.push_back(recvid[p]);
    }
    void report() {
      if(printid == ROOT) {
        printf("BCAST report:\n");
        printf("sendbuf: %p\n", sendbuf);
        printf("sendoffset: %lu\n", sendoffset);
        printf("recvbuf: %p\n", recvbuf);
        printf("recvoffset: %lu\n", recvoffset);
        printf("count: %lu\n", count);
        printf("sendid %d\n", sendid);
        printf("recvid:\n");
        for(auto index : recvid)
          printf("%d\n", index);
      }
    }
  };

  template <typename T>
  void bcast_tree(const MPI_Comm comm_mpi, int numlevel, int groupsize[], CommBench::library lib[], BCAST<T> bcast, std::vector<CommBench::Comm<T>> commlist[], int level, T *recvbuf) {

    int myid;
    int numproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);

    printid = myid;

    if(numproc != groupsize[0]) {
      printf("ERROR!!! groupsize[0] must be equal to numproc.\n");
      return;
    }
    if(level == numlevel) {
      if(printid == ROOT)
        printf("************************************ leaf level %d groupsize %d\n", level, groupsize[level - 1]);
      // commlist[numlevel - 1].push_back(CommBench::Comm<T>(comm_mpi, lib[numlevel-1]));
      CommBench::Comm<T> comm(comm_mpi, lib[numlevel-1]);
      for(auto recvid : bcast.recvid) {
        int sendid = bcast.sendid;
        // commlist[numlevel - 1].back().add(recvbuf, 0, bcast.recvbuf, bcast.recvoffset, bcast.count, sendid, recvid);
        comm.add(recvbuf, 0, bcast.recvbuf, bcast.recvoffset, bcast.count, sendid, recvid);
        if(printid == ROOT)
          printf("sendid %d recvid: %d\n", sendid, recvid);
      }
      return;
    }

    int numgroup = numproc / groupsize[level];
    int mygroup = myid / groupsize[level];
    int sendgroup = bcast.sendid / groupsize[level];
    if(myid == ROOT)
      printf("level %d, numgroup %d groupsize %d sendgroup %d \n", level, numgroup, groupsize[level], sendgroup);

    for(int group = 0; group < numgroup; group++) {
      std::vector<int> recvids;
      for(auto recvid : bcast.recvid) {
        int recvgroup = recvid / groupsize[level];
        if(recvgroup == group) {
          recvids.push_back(recvid);
          if(myid == ROOT)
            printf("recvid %d recvgroup %d\n", recvid, recvgroup);
        }
      }
      if(recvids.size()) {
        int sendid_new = group * groupsize[level] + bcast.sendid % groupsize[level];
        if(myid == sendid_new && recvbuf == NULL) {
          printf("^^^^^^^^^^^^^^^^^^^^^^^ myid %d allocates recvbuf\n", myid);
          hipMalloc(&recvbuf, bcast.count * sizeof(T));
        }
        BCAST<T> bcast_new(recvbuf, 0, bcast.recvbuf, bcast.recvoffset, bcast.count, sendid_new, recvids.size(), recvids.data());
        bcast_new.report();
        bcast_tree(comm_mpi, numlevel, groupsize, lib, bcast_new, commlist, level + 1, recvbuf);
      } 
    }

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

