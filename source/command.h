
  template <typename T>
  class Command {

    public:

    CommBench::Comm<T> *comm = nullptr;
    ExaComm::Compute<T> *compute = nullptr;

    // COMMUNICATION
    // Command(CommBench::Comm<T> *comm) : comm(comm) {}
    // COMPUTATION
    // Command(ExaComm::Compute<T> *compute) : compute(compute) {}
    // COMMUNICATION + COMPUTATION
    Command(CommBench::Comm<T> *comm, ExaComm::Compute<T> *compute) : comm(comm), compute(compute) {}

    void measure(int warmup, int numiter, size_t count) {
      int myid;
      MPI_Comm_rank(comm_mpi, &myid);

      int numcomm = 0;
      int numcomp = 0;
      MPI_Allreduce(&(comm->numsend), &numcomm, 1, MPI_INT, MPI_SUM, comm_mpi);
      MPI_Allreduce(&(comm->numrecv), &numcomm, 1, MPI_INT, MPI_SUM, comm_mpi);
      MPI_Allreduce(&(compute->numcomp), &numcomp, 1, MPI_INT, MPI_SUM, comm_mpi);
      if(numcomm) {
        if(myid == printid) {
          if(compute->numcomp) printf("COMMAND TYPE: COMMUNICATION + COMPUTATION\n");
          else                 printf("COMMAND TYPE: COMMUNICATION\n");
        }
        comm->measure(warmup, numiter, count);
        if(numcomp)
          compute->measure(warmup, numiter, count);
      }
      else if(numcomp) {
        if(myid == printid)
          printf("COMMAND TYPE: COMPUTATION\n");
        compute->measure(warmup, numiter, count);
      }
    }
  };

  template <typename T>
  void implement(std::vector<std::list<Coll<T>*>> &coll_batch, std::vector<std::list<Command<T>>> &pipeline, int pipeoffset) {

    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    for(auto &coll : coll_batch[0])
        coll->report();

    // REPORT MEMORY
    {
      double buffsize_tot = buffsize * sizeof(T) / 1.e9;
      double recycle_tot = recycle * sizeof(T) / 1.e9;
      double reuse_tot = reuse * sizeof(T) / 1.e9;
      MPI_Allreduce(MPI_IN_PLACE, &buffsize_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &recycle_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &reuse_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      if(myid == printid) {
        printf("********************************************\n\n");
        printf("total buffsize: %.2f GB, reuse: %.2f GB, recycle: %.2f GB\n", buffsize_tot, reuse_tot, recycle_tot);
      }
      std::vector<size_t> buffsize_all(numproc);
      std::vector<size_t> recycle_all(numproc);
      std::vector<size_t> reuse_all(numproc);
      MPI_Allgather(&buffsize, sizeof(size_t), MPI_BYTE, buffsize_all.data(), sizeof(size_t), MPI_BYTE, comm_mpi);
      MPI_Allgather(&recycle, sizeof(size_t), MPI_BYTE, recycle_all.data(), sizeof(size_t), MPI_BYTE, comm_mpi);
      MPI_Allgather(&reuse, sizeof(size_t), MPI_BYTE, reuse_all.data(), sizeof(size_t), MPI_BYTE, comm_mpi);
      if(myid == printid) {
        for(int p = 0; p < numproc; p++)
          printf("ExaComm Memory [%d]: %zu bytes (%.2f GB) - %.2f GB reused - %.2f GB recycled\n", p, buffsize_all[p] * sizeof(T), buffsize_all[p] * sizeof(T) / 1.e9, reuse_all[p] * sizeof(T) / 1.e9, recycle_all[p] * sizeof(T) / 1.e9);
        printf("coll_batch size %zu: ", coll_batch.size());
        for(int i = 0; i < coll_batch.size(); i++)
          printf("%zu ", coll_batch[i].size());
        printf("\n\n");
      }
    }

    std::vector<std::list<Coll<T>*>> coll_pipeline;
    std::vector<Coll<T>*> coll_mixed;

    std::vector<int> lib;
    std::vector<int> lib_hash(CommBench::numlib);
    {
      for(int i = 0; i < coll_batch.size(); i++) {
        for(auto &coll : coll_batch[i])
          lib_hash[coll->lib]++;
        for(int j = 0; j < i * pipeoffset; j++)
          coll_batch[i].push_front(new ExaComm::Coll<T>(CommBench::MPI));
      }
      for(int i = 0; i < CommBench::numlib; i++)
        if(lib_hash[i]) {
          lib_hash[i] = lib.size();
          lib.push_back(i);
          pipeline.push_back(std::list<Command<T>>());
          coll_pipeline.push_back(std::list<Coll<T>*>());
        }
    }

    // REPORT DEGERATE PIPELINE
    report_pipeline(coll_batch);

    {
      using Iter = typename std::list<ExaComm::Coll<T>*>::iterator;
      std::vector<Iter> coll_ptr(coll_batch.size());
      for(int i = 0; i < coll_batch.size(); i++)
        coll_ptr[i] = coll_batch[i].begin();
      while(true) {
        bool finished = true;
        for(int i = 0; i < coll_batch.size(); i++)
          if(coll_ptr[i] != coll_batch[i].end())
            finished = false;
        if(finished)
          break;
        ExaComm::Coll<T> *coll_total = new ExaComm::Coll<T>(CommBench::null);
        std::vector<ExaComm::Coll<T>*> coll_temp(lib.size());
        std::vector<CommBench::Comm<T>*> comm_temp(lib.size());
        std::vector<ExaComm::Compute<T>*> compute_temp(lib.size());
        for(int i = 0; i < lib.size(); i++) {
          coll_temp[i] = new ExaComm::Coll<T>((CommBench::library) lib[i]);
          comm_temp[i] = new CommBench::Comm<T>((CommBench::library) lib[i]);
          compute_temp[i] = new ExaComm::Compute<T>();
        }
        for(int i = 0; i < coll_batch.size(); i++)
          if(coll_ptr[i] != coll_batch[i].end()) {
            ExaComm::Coll<T> *coll = *coll_ptr[i];
            coll_ptr[i]++;
            for(int i = 0; i < coll->numcomm; i++) {
              coll_total->add(coll->sendbuf[i], coll->sendoffset[i], coll->recvbuf[i], coll->recvoffset[i], coll->count[i], coll->sendid[i], coll->recvid[i]);
              coll_temp[lib_hash[coll->lib]]->add(coll->sendbuf[i], coll->sendoffset[i], coll->recvbuf[i], coll->recvoffset[i], coll->count[i], coll->sendid[i], coll->recvid[i]);
              comm_temp[lib_hash[coll->lib]]->add(coll->sendbuf[i], coll->sendoffset[i], coll->recvbuf[i], coll->recvoffset[i], coll->count[i], coll->sendid[i], coll->recvid[i]);
            }
            for(int i = 0; i < coll->numcompute; i++) {
              coll_total->add(coll->inputbuf[i], coll->outputbuf[i], coll->numreduce[i], coll->compid[i]);
              coll_temp[lib_hash[coll->lib]]->add(coll->inputbuf[i], coll->outputbuf[i], coll->numreduce[i], coll->compid[i]);
              compute_temp[lib_hash[coll->lib]]->add(coll->inputbuf[i], coll->outputbuf[i], coll->numreduce[i], coll->compid[i]);
            }
          }
        if(coll_total->numcomm + coll_total->numcompute) {
          for(int i = 0; i < lib.size(); i++) {
            coll_pipeline[i].push_back(coll_temp[i]);
            pipeline[i].push_back(Command<T>(comm_temp[i], compute_temp[i]));
          }
          coll_mixed.push_back(coll_total);
        }
        else {
          delete coll_total;
          for(int i = 0; i < lib.size(); i++) {
            delete coll_temp[i];
            delete comm_temp[i];
            delete compute_temp[i];
          }
        }
      }
    }

    // REPORT MIXED PIPELINE
    for(int i = 0; i < coll_mixed.size(); i++)
      if(i < coll_batch[0].size() || i >= coll_mixed.size() - coll_batch[0].size()) {
        if(myid == printid)
          printf("MIXED (OVERLAPPED) STEP: %d / %ld\n", i, coll_mixed.size());
        coll_mixed[i]->report();
      }
    report_pipeline(coll_pipeline);
  }

