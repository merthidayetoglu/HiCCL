
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
  void reduce_tree(const MPI_Comm &comm_mpi, int numlevel, int groupsize[], CommBench::library lib[], std::vector<REDUCE<T>> reducelist, int level, std::list<Command<T>> &commandlist) {

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
    commandlist.push_back(Command<T>(comm));
    commandlist.push_back(Command<T>(compute));

    std::vector<REDUCE<T>> reducelist_new;

    int numgroup = numproc / groupsize[level];

    if(printid == ROOT) {
      printf("level %d groupsize %d numgroup %d\n", level, groupsize[level], numgroup);
    }
    for(auto &reduce : reducelist)
      reduce.report(ROOT);

    // GLOBAL COMMUNICATIONS
    {
      for(auto reduce : reducelist) {
        std::vector<int> sendids_new;
        T* outputbuf;
        size_t outputoffset;
        int recvgroup = reduce.recvid / groupsize[level];
        for(int sendgroup = 0; sendgroup < numgroup; sendgroup++) {
          std::vector<int> sendids;
          // FIND OUTBOUND COMMUNICATIONS
          for(auto &sendid : reduce.sendids) {
            if(sendid / groupsize[level] == sendgroup)
              sendids.push_back(sendid);
          }
          if(sendids.size()) {
            if(printid == ROOT) {
              printf("recvgroup: %d recvid: %d sendgroup: %d sendids: ", recvgroup, reduce.recvid, sendgroup);
              for(auto sendid : sendids)
                printf("%d ", sendid);
              printf("\n");
            }
            int recvid = sendgroup * groupsize[level] + reduce.recvid % groupsize[level];
            if(myid == recvid) {
              if(level == 0) {
                outputbuf = reduce.recvbuf;
                outputoffset = reduce.recvoffset;
              }
              else {
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
            std::vector<T*> inputbuf;
            for(auto &sendid : sendids) {
              if(sendid != recvid) {
                T *recvbuf;
                if(myid == recvid) {
#ifdef PORT_CUDA
                  cudaMalloc(&recvbuf, reduce.count * sizeof(T));
#elif defined PORT_HIP
                  hipMalloc(&recvbuf, reduce.count * sizeof(T));
#else
                  recvbuf = new T[reduce.count];
#endif
                  buffsize += reduce.count;
                }
                comm->add(reduce.sendbuf, reduce.sendoffset, recvbuf, 0, reduce.count, sendid, recvid);
                inputbuf.push_back(recvbuf);
              }
              else
                inputbuf.push_back(reduce.sendbuf + reduce.sendoffset);
            }
            compute->add(inputbuf, outputbuf + outputoffset, reduce.count, recvid);
            sendids_new.push_back(recvid);
          }
  	}
        reducelist_new.push_back(REDUCE<T>(outputbuf, outputoffset, reduce.recvbuf, reduce.recvoffset, reduce.count, sendids_new, reduce.recvid));
      }
    }
    reduce_tree(comm_mpi, numlevel, groupsize, lib, reducelist_new, level - 1, commandlist);
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
