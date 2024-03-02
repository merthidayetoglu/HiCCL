
  template <typename T>
  struct BROADCAST {
    public:
    T* sendbuf;
    size_t sendoffset;
    T* recvbuf;
    size_t recvoffset;
    size_t count;
    int sendid;
    std::vector<int> recvids;

    void report() {
      if(printid < 0)
        return;
      if(myid == sendid) {
        MPI_Send(&sendbuf, sizeof(T*), MPI_BYTE, printid, 0, comm_mpi);
        MPI_Send(&sendoffset, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
      }
      for(auto &recvid : this->recvids) {
        if(myid == recvid) {
          MPI_Send(&recvbuf, sizeof(T*), MPI_BYTE, printid, 0, comm_mpi);
          MPI_Send(&recvoffset, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
        }
      }
      if(myid == printid) {
        T* sendbuf_sendid;
        size_t sendoffset_sendid;
        MPI_Recv(&sendbuf_sendid, sizeof(T*), MPI_BYTE, sendid, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&sendoffset_sendid, sizeof(size_t), MPI_BYTE, sendid, 0, comm_mpi, MPI_STATUS_IGNORE);
	std::vector<T*> recvbuf_recvid(recvids.size());
	std::vector<size_t> recvoffset_recvid(recvids.size());
        for(int recv = 0; recv < recvids.size(); recv++) {
          MPI_Recv(&recvbuf_recvid[recv], sizeof(T*), MPI_BYTE, recvids[recv], 0, comm_mpi, MPI_STATUS_IGNORE);
          MPI_Recv(&recvoffset_recvid[recv], sizeof(size_t), MPI_BYTE, recvids[recv], 0, comm_mpi, MPI_STATUS_IGNORE);
        }
        printf("BROADCAST report: count %lu (", count);
        CommBench::print_data(count * sizeof(T));
        printf(")\n");
        char text[1000];
        int n = sprintf(text, "sendid %d sendbuf %p sendoffset %lu -> ", sendid, sendbuf_sendid, sendoffset_sendid);
        printf("%s", text);
        memset(text, ' ', n);
        for(int recv = 0; recv < recvids.size(); recv++) {
          printf("recvid: %d recvbuf %p recvoffset %lu\n", recvids[recv], recvbuf_recvid[recv], recvoffset_recvid[recv]);
          printf("%s", text);
        }
        printf("\n");
      }
    }

    BROADCAST(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, std::vector<int> &recvids) : sendbuf(sendbuf), sendoffset(sendoffset), recvbuf(recvbuf), recvoffset(recvoffset), count(count), sendid(sendid), recvids(recvids) { }

    BROADCAST(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) : sendbuf(sendbuf), sendoffset(sendoffset), recvbuf(recvbuf), recvoffset(recvoffset), count(count), sendid(sendid) {
      for(int i = 0; i < numproc; i++) {
        if(recvid == numproc)
          recvids.push_back(i);
        else if(recvid == -1) {
          if(i != sendid)
            recvids.push_back(i);
        }
        else
          if(i == recvid)
            recvids.push_back(i);
      }
    }
  };

  template <typename T>
  void bcast_tree(int numlevel, int groupsize[], CommBench::library lib[], std::vector<BROADCAST<T>> bcastlist, int level, std::list<Coll<T>*> &coll_list) {

    if(numproc != groupsize[0]) {
      printf("ERROR!!! groupsize[0] must be equal to numproc.\n");
      return;
    }

    // EXIT CONDITION
    if(bcastlist.size() == 0)
      return;

    Coll<T> *coll_temp = new Coll<T>(lib[level-1]);

    std::vector<BROADCAST<T>> bcastlist_new;

    //  SELF COMMUNICATION
    if(level == numlevel) {
      // if(printid == printid)
      //   printf("************************************ leaf level %d groupsize %d\n", level, groupsize[level - 1]);
      for(auto bcast : bcastlist)
        for(auto recvid : bcast.recvids) {
          coll_temp->add(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, bcast.sendid, recvid);
        }
      // if(printid == printid)
      //   printf("\n");
    }
    else {
      int numgroup = numproc / groupsize[level];
      // LOCAL COMMUNICATIONS
      {
        for(auto bcast : bcastlist) {
          int sendgroup = bcast.sendid / groupsize[level];
          for(int recvgroup = 0; recvgroup < numgroup; recvgroup++) {
            if(sendgroup == recvgroup) {
              std::vector<int> recvids;
              for(auto recvid : bcast.recvids) {
                int group = recvid / groupsize[level];
                if(group == recvgroup)
                  recvids.push_back(recvid);
              }
              if(recvids.size())
                bcastlist_new.push_back(BROADCAST<T>(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, bcast.sendid, recvids));
            }
          }
        }
      }
      // GLOBAL COMMUNICATIONS
      {
        for(int recvgroup = 0; recvgroup < numgroup; recvgroup++) {
          for(auto bcast : bcastlist) {
            int sendgroup = bcast.sendid / groupsize[level];
            if(sendgroup != recvgroup) {
              std::vector<int> recvids;
              for(auto recvid : bcast.recvids) {
                if(recvid / groupsize[level] == recvgroup)
                  recvids.push_back(recvid);
              }
              if(recvids.size()) {
                int recvid = recvgroup * groupsize[level] + bcast.sendid % groupsize[level];
                // if(printid == printid)
                //  printf("level %d groupsize %d numgroup %d sendgroup %d recvgroup %d recvid %d\n", level, groupsize[level], numgroup, sendgroup, recvgroup, recvid);
                T *recvbuf;
                size_t recvoffset;
                bool found = false;
                for(auto it = recvids.begin(); it != recvids.end(); ++it) {
                  if(*it == recvid) {
                    //if(printid == printid)
                    //  printf("******************************************************************************************* found recvid %d\n", *it);
                    recvbuf = bcast.recvbuf;
                    recvoffset = bcast.recvoffset;
                    found = true;
                    recvids.erase(it);
                    break;
                  }
                }
                if(found) {
                  if(myid == recvid)
                    reuse += bcast.count;
                }
                else {
                  if(myid == recvid) {
                    CommBench::allocate(recvbuf, bcast.count);
                    recvoffset = 0;
                    buffsize += bcast.count;
                  }
                  //if(printid == printid)
                  //  printf("^^^^^^^^^^^^^^^^^^^^^^^ recvid %d myid %d allocates\n", recvid, myid);
                }
                coll_temp->add(bcast.sendbuf, bcast.sendoffset, recvbuf,  recvoffset, bcast.count, bcast.sendid, recvid);
                if(recvids.size())
                  bcastlist_new.push_back(BROADCAST<T>(recvbuf, recvoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, recvid, recvids));
              }
            }
          }
        }
      }
    }
    if(coll_temp->numcomm)
      coll_list.push_back(coll_temp);
    else
      delete coll_temp;
    bcast_tree(numlevel, groupsize, lib, bcastlist_new, level + 1, coll_list);
  }

  template<typename T>
  void bcast_ring(int groupsize, CommBench::library lib, std::vector<BROADCAST<T>> &bcastlist, std::vector<BROADCAST<T>> &bcastlist_intra, std::list<Coll<T>*> &coll_list) {

    std::vector<BROADCAST<T>> bcastlist_extra;

    Coll<T> *coll_temp = new Coll<T>(lib);

    for(auto &bcast : bcastlist) {
      int sendnode = bcast.sendid / groupsize;
      std::vector<int> recvids_intra;
      std::vector<int> recvids_extra;
      for(auto &recvid : bcast.recvids) {
        int recvnode = recvid / groupsize;
        if(sendnode == recvnode)
          recvids_intra.push_back(recvid);
        else
          recvids_extra.push_back(recvid);
      }
      // if(printid == printid)
      //   printf("recvids_intra: %zu recvids_extra: %zu\n", recvids_intra.size(), recvids_extra.size());
      if(recvids_intra.size())
        bcastlist_intra.push_back(BROADCAST<T>(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, bcast.sendid, recvids_intra));
      if(recvids_extra.size()) {
        T *recvbuf;
        size_t recvoffset;
        int recvid = ((sendnode + 1) % (numproc / groupsize)) * groupsize + bcast.sendid % groupsize;
        bool found = false;
        for(auto it = recvids_extra.begin(); it != recvids_extra.end(); it++)
          if(*it == recvid) {
            found = true;
            recvids_extra.erase(it);
            break;
          }
	if(myid == recvid) {
          if(found) {
            recvbuf = bcast.recvbuf;
            recvoffset = bcast.recvoffset;
            reuse += bcast.count;
          }
          else {
            CommBench::allocate(recvbuf, bcast.count);
            recvoffset = 0;
            buffsize += bcast.count;
          }
        }
        coll_temp->add(bcast.sendbuf, bcast.sendoffset, recvbuf, recvoffset, bcast.count, bcast.sendid, recvid);
        if(recvids_extra.size())
          bcastlist_extra.push_back(BROADCAST<T>(recvbuf, recvoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, recvid, recvids_extra));
      }
    }
    if(coll_temp->numcomm)
      coll_list.push_back(coll_temp);
    else
      delete coll_temp;

    if(bcastlist_extra.size()) // IMPLEMENT RING FOR EXTRA-NODE COMMUNICATIONS (IF THERE IS STILL LEFT)
      bcast_ring(groupsize, lib, bcastlist_extra, bcastlist_intra, coll_list);
    /*else { // ELSE IMPLEMENT TREE FOR INTRA-NODE COMMUNICATION
      std::vector<int> groupsize_temp(groupsize, groupsize + numlevel);
      groupsize_temp[0] = numproc;
      bcast_tree(numlevel, groupsize_temp.data(), lib, bcastlist_intra, 1, coll_list);
    }*/
  }

  template <typename T, typename P>
  void stripe(int numstripe, std::vector<BROADCAST<T>> &bcastlist, std::vector<P> &split_list) {

    int nodesize = numstripe;

    // SEPARATE INTRA AND INTER NODES
    std::vector<BROADCAST<T>> bcastlist_intra;
    std::vector<BROADCAST<T>> bcastlist_inter;
    for(auto &bcast : bcastlist) {
      int sendid = bcast.sendid;
      std::vector<int> recvid_intra;
      std::vector<int> recvid_inter;
      for(auto &recvid : bcast.recvids)
        if(sendid / nodesize == recvid / nodesize)
          recvid_intra.push_back(recvid);
        else
          recvid_inter.push_back(recvid);
      if(recvid_inter.size())
        bcastlist_inter.push_back(BROADCAST<T>(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, bcast.sendid, bcast.recvids));
      else
        bcastlist_intra.push_back(BROADCAST<T>(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, bcast.sendid, bcast.recvids));
    }
    // CLEAR BROADCASTLIST
    bcastlist.clear();
    // ADD INTRA-NODE BROADCAST DIRECTLY (IF ANY)
    for(auto &bcast : bcastlist_intra)
      bcastlist.push_back(BROADCAST<T>(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, bcast.sendid, bcast.recvids));

    // ADD INTER-NODE BROADCAST BY STRIPING
    if(bcastlist_inter.size()) {
      for(auto &bcast : bcastlist_inter) {
        int sendgroup = bcast.sendid / nodesize;
        size_t splitoffset = 0;
        for(int stripe = 0; stripe < numstripe; stripe++) {
          int sender = sendgroup * nodesize + stripe;
          size_t splitcount = bcast.count / numstripe + (stripe < bcast.count % numstripe ? 1 : 0);
          if(splitcount) {
            T *sendbuf;
            size_t sendoffset;
            std::vector<int> recvids = bcast.recvids;
            if(sender != bcast.sendid) {
              bool found = false;
              // REUSE
              for(auto it = recvids.begin(); it < recvids.end(); it++) {
                if(*it == sender) {
                  recvids.erase(it);
                  found = true;
                  break;
                }
              }
              if(found) {
                if(myid == sender) {
                  sendbuf = bcast.recvbuf;
                  sendoffset = bcast.recvoffset + splitoffset;
                  reuse += splitcount;
                }
              }
              else {
                if(myid == sender) {
                  CommBench::allocate(sendbuf, splitcount);
                  sendoffset = 0;
                  buffsize += splitcount;
                }
              }
              split_list.push_back(P(bcast.sendbuf, bcast.sendoffset + splitoffset, sendbuf, sendoffset, splitcount, bcast.sendid, sender));
            }
            else {
              if(myid == sender) {
                sendbuf = bcast.sendbuf;
                sendoffset = bcast.sendoffset + splitoffset;
                reuse += splitcount;
              }
            }
            bcastlist.push_back(BROADCAST<T>(sendbuf, sendoffset, bcast.recvbuf, bcast.recvoffset + splitoffset, splitcount, sender, recvids));
            splitoffset += splitcount;
          }
          else
            break;
        }
      }
    }
  }

  template <typename T>
  void partition(std::vector<BROADCAST<T>> &bcastlist, int numbatch, std::vector<std::vector<BROADCAST<T>>> &bcast_batch) {
    for(auto &bcast : bcastlist) {
      size_t batchoffset = 0;
      for(int batch = 0; batch < numbatch; batch++) {
        size_t batchsize = bcast.count / numbatch + (batch < bcast.count % numbatch ? 1 : 0);
        if(batchsize) {
          bcast_batch[batch].push_back(BROADCAST<T>(bcast.sendbuf, bcast.sendoffset + batchoffset, bcast.recvbuf, bcast.recvoffset + batchoffset, batchsize, bcast.sendid, bcast.recvids));
          batchoffset += batchsize;
        }
        else
          break;
      }
    }
  }
