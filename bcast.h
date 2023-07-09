
template <typename T>
  struct BCAST {
    public:
    T* const sendbuf;
    const size_t sendoffset;
    T* const recvbuf;
    const size_t recvoffset;
    const size_t count;
    const int sendid;
    std::vector<int> recvids;

    BCAST(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, std::vector<int> &recvids)
    : sendbuf(sendbuf), sendoffset(sendoffset), recvbuf(recvbuf), recvoffset(recvoffset), count(count), sendid(sendid), recvids(recvids) {}

    void report(int id) {
      if(printid == sendid) {
        MPI_Send(&sendbuf, sizeof(T*), MPI_BYTE, id, 0, MPI_COMM_WORLD);
        MPI_Send(&sendoffset, sizeof(size_t), MPI_BYTE, id, 0, MPI_COMM_WORLD);
      }
      for(auto recvid : this->recvids)
        if(printid == recvid) {
          MPI_Send(&recvbuf, sizeof(T*), MPI_BYTE, id, 0, MPI_COMM_WORLD);
          MPI_Send(&recvoffset, sizeof(size_t), MPI_BYTE, id, 0, MPI_COMM_WORLD);
        }
      if(printid == id) {
        T* sendbuf_sendid;
        size_t sendoffset_sendid;
        MPI_Recv(&sendbuf_sendid, sizeof(T*), MPI_BYTE, sendid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&sendoffset_sendid, sizeof(size_t), MPI_BYTE, sendid, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        size_t recvoffset_recvid[recvids.size()];
        T* recvbuf_recvid[recvids.size()];
        for(int recv = 0; recv < recvids.size(); recv++) {
          MPI_Recv(recvbuf_recvid + recv, sizeof(T*), MPI_BYTE, recvids[recv], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(recvoffset_recvid + recv, sizeof(size_t), MPI_BYTE, recvids[recv], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        printf("BCAST report: count %lu\n", count);
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

  #define FACTOR_LEVEL
  // #define FACTOR_LOCAL
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
        for(auto recvid : bcast.recvids) {
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
            for(auto recvid : bcast.recvids) {
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
            for(auto recvid : bcast.recvids) {
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
                buffsize += bcast.count;
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

  template <typename T>
  void stripe(const MPI_Comm &comm_mpi, int numlevel, int groupsize[], CommBench::library lib[], std::vector<BCAST<T>> &bcastlist, std::list<CommBench::Comm<T>*> &commlist, std::list<Command<T>> &commandlist) {

    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    // SEPARATE INTRA AND INTER NODES
    std::vector<BCAST<T>> bcastlist_intra;
    std::vector<BCAST<T>> bcastlist_inter;
    for(auto &bcast : bcastlist) {
      int sendid = bcast.sendid;
      std::vector<int> recvid_intra;
      std::vector<int> recvid_inter;
      for(auto &recvid : bcast.recvids)
        if(sendid / groupsize[0] == recvid / groupsize[0])
          recvid_intra.push_back(recvid);
        else
          recvid_inter.push_back(recvid);
      if(recvid_inter.size())
        bcastlist_inter.push_back(BCAST<T>(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, bcast.sendid, bcast.recvids));
      else
        bcastlist_intra.push_back(BCAST<T>(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, bcast.sendid, bcast.recvids));
    }
    if(printid == ROOT) {
      printf("broadcast striping group size: %d numgroups: %d\n", groupsize[0], numproc / groupsize[0]);
      printf("number of original broadcasts: %zu\n", bcastlist.size());
      printf("number of intra-node broadcast: %zu number of inter-node broadcast: %zu\n", bcastlist_intra.size(), bcastlist_inter.size());
      printf("\n");
    }

    // CLEAR BCASTLIST
    bcastlist.clear();

    // ADD INTRA-NODE BROADCAST DIRECTLY (IF ANY)
    // bcastlist.insert( bcastlist.end(), bcastlist_intra.begin(), bcastlist_intra.end() );
    for(auto bcast : bcastlist_intra)
      bcastlist.push_back(BCAST<T>(bcast.sendbuf, bcast.sendoffset, bcast.recvbuf, bcast.recvoffset, bcast.count, bcast.sendid, bcast.recvids));

    // ADD INTER-NODE BROADCAST BY STRIPING
    if(bcastlist_inter.size())
    {
      int stripelevel = -1;
      for(int level = numlevel - 1; level > -1; level--)
        if(groupsize[level] >= groupsize[0]) {
           stripelevel = level;
           break;
        }
      if(printid == ROOT)
        printf("stripe level: %d\n", stripelevel);
      std::vector<CommBench::Comm<T>*> split;
      for(int level = stripelevel; level < numlevel; level++)
        split.push_back(new CommBench::Comm<T>(comm_mpi, lib[level]));
      for(auto &bcast : bcastlist_inter) {
        int sendgroup = bcast.sendid / groupsize[0];
        int mygroup = myid / groupsize[0];
        T *sendbuf_temp;
        size_t splitcount = bcast.count / groupsize[0];
        if(mygroup == sendgroup && myid != bcast.sendid) {
#ifdef PORT_CUDA
          cudaMalloc(&sendbuf_temp, splitcount * sizeof(T));
#elif defined PORT_HIP
          hipMalloc(&sendbuf_temp, splitcount * sizeof(T));
#endif
          buffsize += splitcount;
        }
        // SPLIT
        for(int p = 0; p < groupsize[0]; p++) {
          int recver = sendgroup * groupsize[0] + p;
          if(printid == ROOT)
            printf("split ");
          if(recver != bcast.sendid) {
            for(int level = numlevel - 1; level > 0; level--) {
              if(recver / groupsize[level] == bcast.sendid / groupsize[level]) {
                if(printid == ROOT)
                  printf("level %d ", level);
                split[level - stripelevel]->add(bcast.sendbuf, bcast.sendoffset + p * splitcount, sendbuf_temp, 0, splitcount, bcast.sendid, recver);
                break;
              }
            }
            bcastlist.push_back(BCAST<T>(sendbuf_temp, 0, bcast.recvbuf, bcast.recvoffset + p * splitcount, splitcount, recver, bcast.recvids));
          }
          else {
            if(printid == ROOT)
              printf("* skip self\n");
            bcastlist.push_back(BCAST<T>(bcast.sendbuf, bcast.sendoffset + p * splitcount, bcast.recvbuf, bcast.recvoffset + p * splitcount, splitcount, bcast.sendid, bcast.recvids));
          }
        }
      }
      commlist.insert( commlist.end(), split.begin(), split.end() );
    }
  }
