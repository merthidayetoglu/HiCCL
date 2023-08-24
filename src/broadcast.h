
  template <typename T>
  struct BROADCAST {
    public:
    T* const sendbuf;
    const size_t sendoffset;
    T* const recvbuf;
    const size_t recvoffset;
    const size_t count;
    const int sendid;
    std::vector<int> recvids;

    BROADCAST(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, std::vector<int> &recvids)
    : sendbuf(sendbuf), sendoffset(sendoffset), recvbuf(recvbuf), recvoffset(recvoffset), count(count), sendid(sendid), recvids(recvids) {}
    BROADCAST(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid)
    : sendbuf(sendbuf), sendoffset(sendoffset), recvbuf(recvbuf), recvoffset(recvoffset), count(count), sendid(sendid) { recvids.push_back(recvid); }

    void report(int id) {
      if(printid == sendid) {
        MPI_Send(&sendbuf, sizeof(T*), MPI_BYTE, id, 0, MPI_COMM_WORLD);
        MPI_Send(&sendoffset, sizeof(size_t), MPI_BYTE, id, 0, MPI_COMM_WORLD);
      }
      for(auto &recvid : this->recvids) {
        if(printid == recvid) {
          MPI_Send(&recvbuf, sizeof(T*), MPI_BYTE, id, 0, MPI_COMM_WORLD);
          MPI_Send(&recvoffset, sizeof(size_t), MPI_BYTE, id, 0, MPI_COMM_WORLD);
        }
      }
      if(printid == id) {
        T* sendbuf_sendid;
        size_t sendoffset_sendid;
        MPI_Recv(&sendbuf_sendid, sizeof(T*), MPI_BYTE, sendid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&sendoffset_sendid, sizeof(size_t), MPI_BYTE, sendid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	std::vector<T*> recvbuf_recvid(recvids.size());
	std::vector<size_t> recvoffset_recvid(recvids.size());
        for(int recv = 0; recv < recvids.size(); recv++) {
          MPI_Recv(recvbuf_recvid.data() + recv, sizeof(T*), MPI_BYTE, recvids[recv], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(recvoffset_recvid.data() + recv, sizeof(size_t), MPI_BYTE, recvids[recv], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        printf("BROADCAST report: count %lu\n", count);
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
  };

  template <typename T>
  void bcast_tree(const MPI_Comm &comm_mpi, int numlevel, int groupsize[], CommBench::library lib[], std::vector<BROADCAST<T>> bcastlist, int level, std::list<Command<T>> &commandlist) {

    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    if(numproc != groupsize[0]) {
      printf("ERROR!!! groupsize[0] must be equal to numproc.\n");
      return;
    }

    // EXIT CONDITION
    if(bcastlist.size() == 0)
      return;

    CommBench::Comm<T> *comm_temp = new CommBench::Comm<T>(comm_mpi, lib[level-1]);
    bool commfound = false;

    std::vector<BROADCAST<T>> bcastlist_new;

    //  SELF COMMUNICATION
    if(level == numlevel) {
      if(printid == ROOT)
         printf("************************************ leaf level %d groupsize %d\n", level, groupsize[level - 1]);
      for(auto bcast : bcastlist)
        for(auto recvid : bcast.recvids) {
          comm_temp->add(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, bcast.sendid, recvid);
          commfound = true;
        }
      if(printid == ROOT)
        printf("\n");
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
                // if(printid == ROOT)
                //  printf("level %d groupsize %d numgroup %d sendgroup %d recvgroup %d recvid %d\n", level, groupsize[level], numgroup, sendgroup, recvgroup, recvid);
                T *recvbuf;
                size_t recvoffset;
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
                if(myid == recvid) {
                  if(found)
                    reuse += bcast.count;
                  else {
#ifdef PORT_CUDA
                    cudaMalloc(&recvbuf, bcast.count * sizeof(T));
#elif defined PORT_HIP
                    hipMalloc(&recvbuf, bcast.count * sizeof(T));
#endif
                    buffsize += bcast.count;
                    recvoffset = 0;
                    printf("^^^^^^^^^^^^^^^^^^^^^^^ recvid %d myid %d allocates recvbuf %p equal %d\n", recvid, myid, recvbuf, myid == recvid);
                  }
                }
                comm_temp->add(bcast.sendbuf, bcast.sendoffset, recvbuf,  recvoffset, bcast.count, bcast.sendid, recvid);
                commfound = true;
                if(recvids.size())
                  bcastlist_new.push_back(BROADCAST<T>(recvbuf, recvoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, recvid, recvids));
              }
            }
          }
        }
      }
    }
    if(commfound)
      commandlist.push_back(Command<T>(comm_temp));
    else
      delete comm_temp;
    bcast_tree(comm_mpi, numlevel, groupsize, lib, bcastlist_new, level + 1, commandlist);
  }

  template<typename T>
  void bcast_ring(const MPI_Comm &comm_mpi, int numlevel, int groupsize[], CommBench::library lib[], std::vector<BROADCAST<T>> &bcastlist, std::vector<BROADCAST<T>> &bcastlist_intra, std::list<Command<T>> &commandlist) {

    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    std::vector<BROADCAST<T>> bcastlist_extra;

    CommBench::Comm<T> *comm_temp = new CommBench::Comm<T>(comm_mpi, lib[0]);
    bool commfound = false;

    for(auto &bcast : bcastlist) {
      int sendnode = bcast.sendid / groupsize[0];
      std::vector<int> recvids_intra;
      std::vector<int> recvids_extra;
      for(auto &recvid : bcast.recvids) {
        int recvnode = recvid / groupsize[0];
        if(sendnode == recvnode)
          recvids_intra.push_back(recvid);
        else
          recvids_extra.push_back(recvid);
      }
      if(printid == ROOT)
        printf("recvids_intra: %zu recvids_extra: %zu\n", recvids_intra.size(), recvids_extra.size());
      if(recvids_intra.size())
        bcastlist_intra.push_back(BROADCAST<T>(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, bcast.sendid, recvids_intra));
      if(recvids_extra.size()) {
        T *recvbuf;
        size_t recvoffset;
        int recvid = ((sendnode + 1) % (numproc / groupsize[0])) * groupsize[0] + bcast.sendid % groupsize[0];
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
#ifdef PORT_CUDA
            cudaMalloc(&recvbuf, bcast.count * sizeof(T));
#elif defined PORT_HIP
            hipMalloc(&recvbuf, bcast.count * sizeof(T));
#else
            recvbuf = new T[bcast.count];
#endif
            recvoffset = 0;
            buffsize += bcast.count;
          }
        }
        comm_temp->add(bcast.sendbuf, bcast.sendoffset, recvbuf, recvoffset, bcast.count, bcast.sendid, recvid);
        commfound = true;
        if(recvids_extra.size())
          bcastlist_extra.push_back(BROADCAST<T>(recvbuf, recvoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, recvid, recvids_extra));
      }
    }
    if(commfound)
      commandlist.push_back(Command<T>(comm_temp));
    else
      delete comm_temp;
    if(bcastlist_extra.size())
      // IMPLEMENT RING FOR EXTRA-NODE COMMUNICATIONS (IF THERE IS STILL LEFT)
      bcast_ring(comm_mpi, numlevel, groupsize, lib, bcastlist_extra, bcastlist_intra, commandlist);
    else {
      // ELSE IMPLEMENT TREE FOR INTRA-NODE COMMUNICATION
      std::vector<int> groupsize_temp(groupsize, groupsize + numlevel);
      groupsize_temp[0] = numproc;
      ExaComm::bcast_tree(comm_mpi, numlevel, groupsize_temp.data(), lib, bcastlist_intra, 1, commandlist);
    }
  }

  template <typename T, typename P>
  void stripe(const MPI_Comm &comm_mpi, int numstripe, int stripeoffset, std::vector<BROADCAST<T>> &bcastlist, std::vector<P> &split_list) {

    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    int nodesize = (stripeoffset == 0 ? 1 : numstripe * stripeoffset);

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
    if(printid == ROOT) {
      printf("broadcast numstripe %d stripeoffset %d groupsize: %d numgroups: %d\n", numstripe, stripeoffset, nodesize, numproc / nodesize);
      printf("number of original broadcasts: %zu\n", bcastlist.size());
      printf("number of intra-node broadcast: %zu number of inter-node broadcast: %zu\n", bcastlist_intra.size(), bcastlist_inter.size());
      printf("\n");
    }
    // CLEAR BROADCASTLIST
    bcastlist.clear();
    // ADD INTRA-NODE BROADCAST DIRECTLY (IF ANY)
    for(auto &bcast : bcastlist_intra)
      bcastlist.push_back(BROADCAST<T>(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, bcast.sendid, bcast.recvids));

    // ADD INTER-NODE BROADCAST BY STRIPING
    if(bcastlist_inter.size())
    {
      for(auto &bcast : bcastlist_inter) {
        int sendgroup = bcast.sendid / nodesize;
        size_t splitoffset = 0;
        for(int stripe = 0; stripe < numstripe; stripe++) {
          int recver = sendgroup * nodesize + stripe * stripeoffset;
          size_t splitcount = bcast.count / numstripe + (stripe < bcast.count % numstripe ? 1 : 0);
          if(splitcount) {
            if(recver != bcast.sendid) {
              T *sendbuf_temp;
              if(myid == recver) {
#ifdef PORT_CUDA
                cudaMalloc(&sendbuf_temp, splitcount * sizeof(T));
#elif defined PORT_HIP
                hipMalloc(&sendbuf_temp, splitcount * sizeof(T));
#endif
                buffsize += splitcount;
              }
              bcastlist.push_back(BROADCAST<T>(sendbuf_temp, 0, bcast.recvbuf, bcast.recvoffset + splitoffset, splitcount, recver, bcast.recvids));
              split_list.push_back(P(bcast.sendbuf, bcast.sendoffset + splitoffset, sendbuf_temp, 0, splitcount, bcast.sendid, recver));
            }
            else
              bcastlist.push_back(BROADCAST<T>(bcast.sendbuf, bcast.sendoffset + splitoffset, bcast.recvbuf, bcast.recvoffset + splitoffset, splitcount, bcast.sendid, bcast.recvids));
            splitoffset += splitcount;
          }
          else
            break;
        }
      }
    }
  }

  template <typename T>
  void batch(std::vector<BROADCAST<T>> &bcastlist, int numbatch, std::vector<std::vector<BROADCAST<T>>> &bcast_batch) {
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

