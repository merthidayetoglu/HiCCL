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
    : sendbuf(sendbuf), sendoffset(sendoffset), recvbuf(recvbuf), recvoffset(recvoffset), count(count), sendid(sendid), recvid(recvid) {};
  };

  template <typename T>
  class Comm {

    const MPI_Comm &comm_mpi;

    std::vector<P2P<T>> addlist;
    std::vector<CommBench::Comm<T>> comm_intra;
    std::vector<CommBench::Comm<T>> comm_inter;
    std::vector<CommBench::Comm<T>> comm_split;
    std::vector<CommBench::Comm<T>> comm_merge;

    void start() {
      for(auto split = comm_split.begin(); split != comm_split.end(); ++split)
        split->launch();
      for(auto split = comm_split.rbegin(); split != comm_split.rend(); ++split)
        split->wait();
      for(auto inter = comm_inter.begin(); inter != comm_inter.end(); ++inter)
        inter->launch();
      for(auto intra = comm_intra.begin(); intra != comm_intra.end(); ++intra)
        intra->run();
    }
    void wait() {
      for(auto intra = comm_intra.rbegin(); intra != comm_intra.rend(); ++intra)
        intra->wait();
      for(auto inter = comm_inter.rbegin(); inter != comm_inter.rend(); ++inter)
        inter->wait();
      for(auto merge = comm_merge.begin(); merge != comm_merge.end(); ++merge)
        merge->launch();
      for(auto merge = comm_merge.rbegin(); merge != comm_merge.rend(); ++merge)
        merge->wait();
    }

    void init_striped(int numlevel, int groupsize[], CommBench::library lib[]) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);

      for(int level = 0; level < numlevel; level++)
        comm_intra.push_back(CommBench::Comm<T>(comm_mpi, lib[level]));

      std::vector<P2P<T>> p2p_inter;

      for(auto p2p : addlist) {
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
            printf("level * ");
          p2p_inter.push_back(P2P<T>(p2p.sendbuf, p2p.sendoffset, p2p.recvbuf, p2p.recvoffset, p2p.count, p2p.sendid, p2p.recvid));
          //comm_inter[0].add(p2p.sendbuf, p2p.sendoffset, p2p.recvbuf, p2p.recvoffset, p2p.count, p2p.sendid, p2p.recvid);
	  //
        }
      }
      if(myid == ROOT)
        printf("* to be splitted");
    }

    void init_mixed(int numlevel, int groupsize[], CommBench::library lib[]) {

      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);

      for(int level = 0; level < numlevel; level++)
        comm_intra.push_back(CommBench::Comm<T>(comm_mpi, lib[level]));

      for(auto p2p : addlist) {
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
    }

    public:

    Comm(const MPI_Comm &comm_mpi, CommBench::library lib)
    : comm_mpi(comm_mpi) {
      comm_inter.push_back(CommBench::Comm<T>(comm_mpi, lib));
    }

    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
      addlist.push_back(ExaComm::P2P<T>(sendbuf, sendoffset, recvbuf, recvoffset, count, sendid, recvid));
    }

    void init_flat() {
      init_mixed(0, NULL, NULL);
    }
    void init_mixed(int groupsize, CommBench::library lib) {
      init_mixed(1, &groupsize, &lib);
    };
    void init_mixed(int groupsize_1, int groupsize_2, CommBench::library lib_1, CommBench::library lib_2) {
      int numlevel = 2;
      int groupsize[numlevel] = {groupsize_1, groupsize_2};
      CommBench::library lib[numlevel] = {lib_1, lib_2};
      init_mixed(numlevel, groupsize, lib);
    }

    void run() {
      start();
      wait();
    }
    void measure(int numwarmup, int numiter) {
      for(auto &comm : comm_inter)
        comm.measure(numwarmup, numiter);
      for(auto &comm : comm_split)
        comm.measure(numwarmup, numiter);
      for(auto &comm : comm_intra)
        comm.measure(numwarmup, numiter);
      for(auto &comm : comm_merge)
        comm.measure(numwarmup, numiter);
    }
  };
}

