
  template <typename T>
  struct REDUCE {
    public:
    T* const sendbuf;
    const size_t sendoffset;
    T* const recvbuf;
    const size_t recvoffset;
    const size_t count;
    std::vector<int> sendids;
    const int recvid;

    REDUCE(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, std::vector<int> &sendids, int recvid)
    : sendbuf(sendbuf), sendoffset(sendoffset), recvbuf(recvbuf), recvoffset(recvoffset), count(count), sendids(sendids), recvid(recvid) {}
    REDUCE(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid)
    : sendbuf(sendbuf), sendoffset(sendoffset), recvbuf(recvbuf), recvoffset(recvoffset), count(count), recvid(recvid) { sendids.push_back(sendid); }
    void report(int id) {
      if(printid == recvid) {
        MPI_Send(&recvbuf, sizeof(T*), MPI_BYTE, id, 0, MPI_COMM_WORLD);
        MPI_Send(&recvoffset, sizeof(size_t), MPI_BYTE, id, 0, MPI_COMM_WORLD);
      }
      for(auto &sendid : this->sendids)
        if(printid == sendid) {
          MPI_Send(&sendbuf, sizeof(T*), MPI_BYTE, id, 0, MPI_COMM_WORLD);
          MPI_Send(&sendoffset, sizeof(size_t), MPI_BYTE, id, 0, MPI_COMM_WORLD);
        }
      if(printid == id) {
        T* recvbuf_recvid;
        size_t recvoffset_recvid;
        MPI_Recv(&recvbuf_recvid, sizeof(T*), MPI_BYTE, recvid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&recvoffset_recvid, sizeof(size_t), MPI_BYTE, recvid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	std::vector<T*> sendbuf_sendid(sendids.size());
	std::vector<size_t> sendoffset_sendid(sendids.size());
        for(int send = 0; send < sendids.size(); send++) {
          MPI_Recv(sendbuf_sendid.data() + send, sizeof(T*), MPI_BYTE, sendids[send], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(sendoffset_sendid.data() + send, sizeof(size_t), MPI_BYTE, sendids[send], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        printf("REDUCE report: count %lu\n", count);
        char text[1000];
        int n = sprintf(text, "recvid %d recvbuf %p recvoffset %lu <- ", recvid, recvbuf_recvid, recvoffset_recvid);
        printf("%s", text);
        memset(text, ' ', n);
        for(int send = 0; send < sendids.size(); send++) {
          printf("sendid: %d sendbuf %p sendoffset %lu\n", sendids[send], sendbuf_sendid[send], sendoffset_sendid[send]);
          printf("%s", text);
        }
        printf("\n");
      }
    }
  };

  template <typename T>
  void reduce_tree(const MPI_Comm &comm_mpi, int numlevel, int groupsize[], CommBench::library lib[], std::vector<REDUCE<T>> reducelist, int level, std::list<Command<T>> &commandlist, std::vector<T*> &recvbuf_ptr, int numrecvbuf) {

    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    if(numproc != groupsize[0]) {
      printf("ERROR!!! groupsize[0] must be equal to numproc.\n");
      return;
    }
    if(reducelist.size() == 0)
      return;

    //  EXIT CONDITION
    if(level == -1)
      return;
    
    CommBench::Comm<T> *comm = new CommBench::Comm<T>(comm_mpi, lib[level]);
    ExaComm::Compute<T> *compute = new ExaComm::Compute<T>(comm_mpi);
    bool compute_found = false;
    bool commfound = false;

    std::vector<REDUCE<T>> reducelist_new;

    int numgroup = numproc / groupsize[level];

    if(printid == ROOT) {
      printf("level %d groupsize %d numgroup %d\n", level, groupsize[level], numgroup);
    }
    // for(auto &reduce : reducelist)
    //  reduce.report(ROOT);

    {
      for(auto reduce : reducelist) {
        std::vector<int> sendids_new;
        std::vector<T*> sendbuf_new;
        std::vector<size_t> sendoffset_new;
        int recvgroup = reduce.recvid / groupsize[level];
        for(int sendgroup = 0; sendgroup < numgroup; sendgroup++) {
          std::vector<int> sendids;
          for(auto &sendid : reduce.sendids)
            if(sendid / groupsize[level] == sendgroup)
              sendids.push_back(sendid);
          if(sendids.size()) {
            /*if(printid == ROOT) {
              printf("recvgroup: %d recvid: %d sendgroup: %d sendids: ", recvgroup, reduce.recvid, sendgroup);
              for(auto sendid : sendids)
                printf("%d ", sendid);
              printf("\n");
            }*/
            int recvid = sendgroup * groupsize[level] + reduce.recvid % groupsize[level];
            T* outputbuf;
            size_t outputoffset;
            if(myid == recvid) {
              if(recvid == reduce.recvid) {
                outputbuf = reduce.recvbuf;
                outputoffset = reduce.recvoffset;
	      }
	      else {
                // printf("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ myid %d send malloc %zu\n", myid, reduce.count * sizeof(T));
#ifdef PORT_CUDA
                cudaMalloc(&outputbuf, reduce.count * sizeof(T));
#elif defined PORT_HIP
                hipMalloc(&outputbuf, reduce.count * sizeof(T));
#else
                outputbuf = new T[reduce.count];
#endif
                outputoffset = 0;
                buffsize += reduce.count;
	      }
            }
            if(sendids.size() > 1) {
              std::vector<T*> inputbuf;
              for(auto &sendid : sendids) {
                if(sendid != recvid) {
                  T *recvbuf;
                  if(myid == recvid) {
                    if(numrecvbuf < recvbuf_ptr.size()) {
                      recvbuf = recvbuf_ptr[numrecvbuf]; // recycle memory
                      recycle += reduce.count;
                    }
                    else {
                      // printf("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ myid %d recv malloc %zu\n", myid, reduce.count * sizeof(T));
#ifdef PORT_CUDA
                      cudaMalloc(&recvbuf, reduce.count * sizeof(T));
#elif defined PORT_HIP
                      hipMalloc(&recvbuf, reduce.count * sizeof(T));
#else
                      recvbuf = new T[reduce.count];
#endif
                      recvbuf_ptr.push_back(recvbuf);
                      buffsize += reduce.count;
                    }
                    numrecvbuf++;
                  }
                  comm->add(reduce.sendbuf, reduce.sendoffset, recvbuf, 0, reduce.count, sendid, recvid);
                  commfound = true;
                  inputbuf.push_back(recvbuf);
                }
                else
                  inputbuf.push_back(reduce.sendbuf + reduce.sendoffset);
              }
              compute->add(inputbuf, outputbuf + outputoffset, reduce.count, recvid);
              compute_found = true;
            }
	    else {
              if(sendids[0] != recvid) {
                comm->add(reduce.sendbuf, reduce.sendoffset, outputbuf, outputoffset, reduce.count, sendids[0], recvid);
                commfound = true;
              }
              else {
                if(level == numlevel - 1) {
                  comm->add(reduce.sendbuf, reduce.sendoffset, outputbuf, outputoffset, reduce.count, sendids[0], recvid);
                  commfound = true;
                }
                else {
                  outputbuf = reduce.sendbuf;
                  outputoffset = reduce.sendoffset;
                }
              }
            }
            sendids_new.push_back(recvid);
            sendbuf_new.push_back(outputbuf);
            sendoffset_new.push_back(outputoffset);
          }
        }
        if(sendids_new.size()) {
          T *sendbuf;
          size_t sendoffset;
          for(int i = 0; i < sendids_new.size(); i++)
            if(myid == sendids_new[i]) {
              sendbuf = sendbuf_new[i];
              sendoffset = sendoffset_new[i];
            }
          reducelist_new.push_back(REDUCE<T>(sendbuf, sendoffset, reduce.recvbuf, reduce.recvoffset, reduce.count, sendids_new, reduce.recvid));
        }
      }
    }
    // ADD COMMUNICATION FOLLOWED BY COMPUTE (IF ANY) OTHERWISE CLEAR MEMORY
    if(commfound)
      commandlist.push_back(Command<T>(comm));
    else
      delete comm;
    if(compute_found)
      commandlist.push_back(Command<T>(compute));
    else
      delete compute;
    reduce_tree(comm_mpi, numlevel, groupsize, lib, reducelist_new, level - 1, commandlist, recvbuf_ptr, 0);
  }

  template <typename T>
  void batch(std::vector<REDUCE<T>> &reducelist, int numbatch, std::vector<std::vector<REDUCE<T>>> &reduce_batch) {
    for(auto &reduce : reducelist) {
      size_t batchoffset = 0;
      for(int batch = 0; batch < numbatch; batch++) {
        size_t batchsize = reduce.count / numbatch + (batch < reduce.count % numbatch ? 1 : 0);
        if(batchsize) {
          reduce_batch[batch].push_back(REDUCE<T>(reduce.sendbuf, reduce.sendoffset + batchoffset, reduce.recvbuf, reduce.recvoffset + batchoffset, batchsize, reduce.sendids, reduce.recvid));
          batchoffset += batchsize;
        }
        else
          break;
      }
    }
  }

  template <typename T, typename P>
  void stripe(const MPI_Comm &comm_mpi, int nodesize, std::vector<REDUCE<T>> &reducelist, std::vector<P> &merge_list) {

    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    // SEPARATE INTRA AND INTER NODES
    std::vector<REDUCE<T>> reducelist_intra;
    std::vector<REDUCE<T>> reducelist_inter;
    for(auto &reduce : reducelist) {
      int recvid = reduce.recvid;
      std::vector<int> sendid_intra;
      std::vector<int> sendid_inter;
      for(auto &sendid : reduce.sendids)
        if(sendid / nodesize == recvid / nodesize)
          sendid_intra.push_back(sendid);
        else
          sendid_inter.push_back(sendid);
      if(sendid_inter.size())
        reducelist_inter.push_back(REDUCE<T>(reduce.sendbuf, reduce.sendoffset, reduce.recvbuf, reduce.recvoffset, reduce.count, reduce.sendids, reduce.recvid));
      else
        reducelist_intra.push_back(REDUCE<T>(reduce.sendbuf, reduce.sendoffset, reduce.recvbuf, reduce.recvoffset, reduce.count, reduce.sendids, reduce.recvid));
    }
    if(printid == ROOT) {
      printf("reduction striping groupsize: %d numgroups: %d\n", nodesize, numproc / nodesize);
      printf("number of original reductions: %zu\n", reducelist.size());
      printf("number of intra-node reductions: %zu number of inter-node reductions: %zu\n", reducelist_intra.size(), reducelist_inter.size());
    }
    // CLEAR REDUCELIST
    reducelist.clear();
    // ADD INTRA-NODE REDUCTION DIRECTLY (IF ANY)
    for(auto &reduce : reducelist_intra)
      reducelist.push_back(REDUCE<T>(reduce.sendbuf, reduce.sendoffset, reduce.recvbuf, reduce.recvoffset, reduce.count, reduce.sendids, reduce.recvid));

    // ADD INTER-NODE REDUCTIONS BY STRIPING
    if(reducelist_inter.size())
    {
      for(auto &reduce : reducelist_inter) {
        int recvnode = reduce.recvid / nodesize;
        size_t splitoffset = 0;
        for(int p = 0; p < nodesize; p++) {
          int recver = recvnode * nodesize + p;
          size_t splitcount = reduce.count / nodesize + (p < reduce.count % nodesize ? 1 : 0);
          if(splitcount) {
            T *recvbuf_temp;
            if(myid == recver) {
#ifdef PORT_CUDA
              cudaMalloc(&recvbuf_temp, splitcount * sizeof(T));
#elif defined PORT_HIP
              hipMalloc(&recvbuf_temp, splitcount * sizeof(T));
#endif
              buffsize += splitcount;
            }
            reducelist.push_back(REDUCE<T>(reduce.sendbuf, reduce.sendoffset + splitoffset, recvbuf_temp, 0, splitcount, reduce.sendids, recver));
            merge_list.push_back(P(recvbuf_temp, 0, reduce.recvbuf, reduce.recvoffset + splitoffset, splitcount, recver, reduce.recvid));
            splitoffset += splitcount;
          }
          else
            break;
        }
      }
    }
  }

