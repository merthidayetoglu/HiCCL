
  template <typename T>
  struct REDUCE {
    public:
    T* sendbuf;
    size_t sendoffset;
    T* recvbuf;
    size_t recvoffset;
    size_t count;
    std::vector<int> sendids;
    int recvid;

    void report() {
      if(printid < 0)
        return;
      if(myid == recvid) {
        MPI_Send(&recvbuf, sizeof(T*), MPI_BYTE, printid, 0, comm_mpi);
        MPI_Send(&recvoffset, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
      }
      for(auto &sendid : this->sendids)
        if(myid == sendid) {
          MPI_Send(&sendbuf, sizeof(T*), MPI_BYTE, printid, 0, comm_mpi);
          MPI_Send(&sendoffset, sizeof(size_t), MPI_BYTE, printid, 0, comm_mpi);
        }
      if(myid == printid) {
        T* recvbuf_recvid;
        size_t recvoffset_recvid;
        MPI_Recv(&recvbuf_recvid, sizeof(T*), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
        MPI_Recv(&recvoffset_recvid, sizeof(size_t), MPI_BYTE, recvid, 0, comm_mpi, MPI_STATUS_IGNORE);
        std::vector<T*> sendbuf_sendid(sendids.size());
        std::vector<size_t> sendoffset_sendid(sendids.size());
        for(int send = 0; send < sendids.size(); send++) {
          MPI_Recv(sendbuf_sendid.data() + send, sizeof(T*), MPI_BYTE, sendids[send], 0, comm_mpi, MPI_STATUS_IGNORE);
          MPI_Recv(sendoffset_sendid.data() + send, sizeof(size_t), MPI_BYTE, sendids[send], 0, comm_mpi, MPI_STATUS_IGNORE);
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

    REDUCE(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, std::vector<int> &sendids, int recvid)
    : sendbuf(sendbuf), sendoffset(sendoffset), recvbuf(recvbuf), recvoffset(recvoffset), count(count), sendids(sendids), recvid(recvid) {}

    REDUCE(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) : sendbuf(sendbuf), sendoffset(sendoffset), recvbuf(recvbuf), recvoffset(recvoffset), count(count), recvid(recvid) {
      for(int i = 0; i < numproc; i++) {
        if(sendid == numproc)
          sendids.push_back(i);
        else if(sendid == -1) {
          if(i != recvid)
            sendids.push_back(i);
        }
        else
          if(i == sendid)
            sendids.push_back(i);
      }
      report();
    }
  };

  template <typename T>
  void reduce_tree(int numlevel, int groupsize[], CommBench::library lib[], std::vector<REDUCE<T>> reducelist, int level, std::list<Coll<T>*> &coll_list, std::vector<T*> &recvbuf_ptr, int numrecvbuf) {

    if(numproc != groupsize[0]) {
      printf("ERROR!!! groupsize[0] must be equal to numproc.\n");
      return;
    }
    if(reducelist.size() == 0)
      return;

    //  EXIT CONDITION
    if(level == -1)
      return;
   
    Coll<T> *coll_temp = new Coll<T>(lib[level]); 

    std::vector<REDUCE<T>> reducelist_new;

    int numgroup = numproc / groupsize[level];

    //if(printid == printid) {
    //  printf("level %d groupsize %d numgroup %d\n", level, groupsize[level], numgroup);
    //}
    // for(auto &reduce : reducelist)
    //  reduce.report(printid);

    {
      for(auto reduce : reducelist) {
        std::vector<int> sendids_new;
        std::vector<T*> sendbuf_new;
        std::vector<size_t> sendoffset_new;
        // int recvgroup = reduce.recvid / groupsize[level];
        for(int sendgroup = 0; sendgroup < numgroup; sendgroup++) {
          std::vector<int> sendids;
          for(auto &sendid : reduce.sendids)
            if(sendid / groupsize[level] == sendgroup)
              sendids.push_back(sendid);
          if(sendids.size()) {
            /*if(printid == printid) {
              printf("recvgroup: %d recvid: %d sendgroup: %d sendids: ", recvgroup, reduce.recvid, sendgroup);
              for(auto sendid : sendids)
                printf("%d ", sendid);
              printf("\n");
            }*/
            int recvid = sendgroup * groupsize[level] + reduce.recvid % groupsize[level];
            T* outputbuf;
            size_t outputoffset;
            if(recvid == reduce.recvid) {
              if(myid == recvid) {
                outputbuf = reduce.recvbuf;
                outputoffset = reduce.recvoffset;
                reuse += reduce.count;
              }
              // if(printid == printid)
              //   printf("recvid %d reuses send memory\n", recvid);
	    }
	    else {
              if(myid == recvid) {
                CommBench::allocate(outputbuf, reduce.count);
                outputoffset = 0;
              }
              // if(printid == printid)
              //    printf("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ proc %d send malloc %zu\n", recvid, reduce.count * sizeof(T));
            }
            if(sendids.size() > 1) {
              std::vector<T*> inputbuf;
              for(auto &sendid : sendids) {
                if(sendid != recvid) {
                  T *recvbuf;
                  if(numrecvbuf < recvbuf_ptr.size()) {
                    if(myid == recvid) {
                      recvbuf = recvbuf_ptr[numrecvbuf]; // recycle memory
                      recycle += reduce.count;
                      numrecvbuf++;
                    }
                    //if(printid == printid)
                    //  printf("recvid %d reuses recv memory\n", recvid);
                  }
                  else {
                    if(myid == recvid) {
                      CommBench::allocate(recvbuf, reduce.count);
                      recvbuf_ptr.push_back(recvbuf);
                      numrecvbuf++;
                    }
                    //if(printid == printid)
                    //   printf("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ proc %d recv malloc %zu\n", recvid, reduce.count * sizeof(T));
                  }
                  /// ADD COMMUNICATION
                  coll_temp->add(reduce.sendbuf, reduce.sendoffset, recvbuf, 0, reduce.count, sendid, recvid);
                  inputbuf.push_back(recvbuf);
                }
                else
                  inputbuf.push_back(reduce.sendbuf + reduce.sendoffset);
              }
              // ADD COMPUTATION
              coll_temp->add(inputbuf, outputbuf + outputoffset, reduce.count, recvid);
            }
	    else {
              if(sendids[0] != recvid) {
                /// ADD COMMUNICATION
                coll_temp->add(reduce.sendbuf, reduce.sendoffset, outputbuf, outputoffset, reduce.count, sendids[0], recvid);
              }
              else {
                if(level == numlevel - 1) {
                  /// ADD COMMUNICATION
                  coll_temp->add(reduce.sendbuf, reduce.sendoffset, outputbuf, outputoffset, reduce.count, sendids[0], recvid);
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
    if(coll_temp->numcomm + coll_temp->numcompute)
      coll_list.push_back(coll_temp);
    else
      delete coll_temp;

    reduce_tree(numlevel, groupsize, lib, reducelist_new, level - 1, coll_list, recvbuf_ptr, 0);
  }

  template<typename T>
  void reduce_ring(int numlevel, int groupsize[], CommBench::library lib[], std::vector<REDUCE<T>> &reducelist, std::vector<REDUCE<T>> &reducelist_intra, std::list<Coll<T>*> &coll_list) {

    //if(printid == printid)
   //   printf("number of original reductions %ld\n", reducelist.size());

    // std::vector<REDUCE<T>> reducelist_intra;
    std::vector<REDUCE<T>> reducelist_extra;

    Coll<T> *coll_temp = new Coll<T>(lib[0]);

    //if(printid == printid)
    //  printf("number of original reductions %ld\n", reducelist.size());
    for(auto &reduce : reducelist) {
      //if(printid == printid)
      //  printf("reduce recvid: %d numsend: %ld\n", reduce.recvid, reduce.sendids.size());
      int recvnode = reduce.recvid / groupsize[0];
      std::vector<int> sendids_intra;
      std::vector<int> sendids_extra;
      for(auto &sendid : reduce.sendids) {
        int sendnode = sendid / groupsize[0];
        if(sendnode == recvnode)
          sendids_intra.push_back(sendid);
        else
          sendids_extra.push_back(sendid);
      }
      //if(printid == printid)
      //  printf("recvid %d numsend %ld sendids_intra: %zu sendids_extra: %zu\n", reduce.recvid, reduce.sendids.size(), sendids_intra.size(), sendids_extra.size());
      if(sendids_extra.size()) {
        int numnode = numproc / groupsize[0];
        int sendnode = (numnode + recvnode + 1) % numnode;
        std::vector<std::vector<int>> sendids(numnode);
        for(auto &sendid : reduce.sendids)
          sendids[sendid / groupsize[0]].push_back(sendid);
        int sendid = sendnode * groupsize[0] + reduce.recvid % groupsize[0];
        /*if(printid == printid) {
          printf("****************** recvnode %d recvid %d sendnode %d sendid %d\n", recvnode, reduce.recvid, sendnode, sendid);
          for(int node = 0; node < numnode; node++) {
            printf("for node %d / %d: ", node, numnode);
            for(auto &sendid : sendids[node])
              printf("%d ", sendid);
            printf("\n");
          }
        }*/
        // FOR SENDING NODE
        T *sendbuf;
        size_t sendoffset;
        bool sendreuse = false;
        if(sendids[sendnode].size() == 1)
          if(sendids[sendnode][0] == sendid) {
            sendbuf = reduce.sendbuf;
            sendoffset = reduce.sendoffset;
            sendreuse = true;
            sendids[sendnode].clear();
            //if(printid == printid)
            //  printf("proc %d reuse %ld\n", sendid, reduce.count);
          }
        if(!sendreuse) {
	  if(myid == sendid) {
            CommBench::allocate(sendbuf, reduce.count);
            sendoffset = 0;
          }
          //if(printid == printid)
          //  printf("proc %d allocate %ld\n", sendid, reduce.count);
        }
        std::vector<int> sendids_extra;
        for(int node = 0; node < numnode; node++)
          if(node != recvnode)
            for(auto &sendid: sendids[node])
              sendids_extra.push_back(sendid);
        reducelist_extra.push_back(REDUCE<T>(reduce.sendbuf, reduce.sendoffset, sendbuf, sendoffset, reduce.count, sendids_extra, sendid));
        //if(printid == printid)
        //  printf("recvid %d sendids_intra: %zu sendids_extra: %zu\n", reduce.recvid, sendids_intra.size(), sendids_extra.size());
        // FOR RECIEVING NODE
        T *recvbuf;
        size_t recvoffset;
        if(sendids_intra.size() == 0) {
          recvbuf = reduce.recvbuf;
          recvoffset = reduce.recvoffset;
          reuse += reduce.count;
        }
	else {
          T *recvbuf_intra;
          if(myid == reduce.recvid) {
	    CommBench::allocate(recvbuf, reduce.count);
            recvoffset = 0;
            CommBench::allocate(recvbuf_intra, reduce.count);
          }
          reducelist_intra.push_back(REDUCE<T>(reduce.sendbuf, reduce.sendoffset, recvbuf_intra, 0, reduce.count, sendids_intra, reduce.recvid));
          std::vector<T*> inputbuf = {recvbuf, recvbuf_intra};
          // ADD COMPUTATION
          coll_temp->add(inputbuf, reduce.recvbuf + reduce.recvoffset, reduce.count, reduce.recvid);
          sendids_intra.push_back(reduce.recvid);
        }
        // ADD COMMUNICATION
        coll_temp->add(sendbuf, sendoffset, recvbuf, recvoffset, reduce.count, sendid, reduce.recvid);
      }
      else
        reducelist_intra.push_back(REDUCE<T>(reduce.sendbuf, reduce.sendoffset, reduce.recvbuf, reduce.recvoffset, reduce.count, reduce.sendids, reduce.recvid));
    }
    /*if(printid == printid) {
      printf("intra reductions: %ld extra reductions: %ld\n\n", reducelist_intra.size(), reducelist_extra.size());
    }*/

    if(reducelist_extra.size())
      reduce_ring(numlevel, groupsize, lib, reducelist_extra, reducelist_intra, coll_list);
    else {
      // COMPLETE RING WITH INTRA-NODE TREE REDUCTION
      std::vector<int> groupsize_temp(groupsize, groupsize + numlevel);
      groupsize_temp[0] = numproc;
      std::vector<T*> recvbuff; // for memory recycling
      reduce_tree(numlevel, groupsize_temp.data(), lib, reducelist_intra, numlevel - 1, coll_list, recvbuff, 0);
    }

    if(coll_temp->numcomm + coll_temp->numcompute)
      coll_list.push_back(coll_temp);
    else
      delete coll_temp;
  }

  template <typename T, typename P>
  void stripe(int numstripe, std::vector<REDUCE<T>> &reducelist, std::vector<P> &merge_list) {

    int nodesize = numstripe;

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
        for(int stripe = 0; stripe < numstripe; stripe++) {
          int recver = recvnode * nodesize + stripe;
          size_t splitcount = reduce.count / numstripe + (stripe < reduce.count % numstripe ? 1 : 0);
          if(splitcount) {
            T *recvbuf;
            size_t recvoffset;
            if(recver != reduce.recvid) {
              if(myid == recver) {
                CommBench::allocate(recvbuf, splitcount);
                recvoffset = 0;
              }
              merge_list.push_back(P(recvbuf, recvoffset, reduce.recvbuf, reduce.recvoffset + splitoffset, splitcount, recver, reduce.recvid));
            }
            else
              if(myid == recver) {
                recvbuf = reduce.recvbuf;
                recvoffset = reduce.recvoffset + splitoffset;
              }
            reducelist.push_back(REDUCE<T>(reduce.sendbuf, reduce.sendoffset + splitoffset, recvbuf, recvoffset, splitcount, reduce.sendids, recver));
            splitoffset += splitcount;
          }
          else
            break;
        }
      }
    }
  }

  template <typename T>
  void partition(std::vector<REDUCE<T>> &reducelist, int numbatch, std::vector<std::vector<REDUCE<T>>> &reduce_batch) {
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
