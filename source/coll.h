  template <typename T>
  class Coll {

    public:

    CommBench::library lib;

    // Communication
    int numcomm = 0;
    std::vector<T*> sendbuf;
    std::vector<size_t> sendoffset;
    std::vector<T*> recvbuf;
    std::vector<size_t> recvoffset;
    std::vector<size_t> count;
    std::vector<int> sendid;
    std::vector<int> recvid;

    // Computation
    int numcompute = 0;
    std::vector<std::vector<T*>> inputbuf;
    std::vector<T*> outputbuf;
    std::vector<size_t> numreduce;
    std::vector<int> compid;

    Coll(CommBench::library lib) : lib(lib) {}

    void add(T *sendbuf, size_t sendoffset, T *recvbuf, size_t recvoffset, size_t count, int sendid, int recvid) {
      this->sendbuf.push_back(sendbuf);
      this->sendoffset.push_back(sendoffset);
      this->recvbuf.push_back(recvbuf);
      this->recvoffset.push_back(recvoffset);
      this->count.push_back(count);
      this->sendid.push_back(sendid);
      this->recvid.push_back(recvid);
      numcomm++;
    }

    void add(std::vector<T*> inputbuf, T* outputbuf, size_t numreduce, int compid) {
      this->inputbuf.push_back(inputbuf);
      this->outputbuf.push_back(outputbuf);
      this->numreduce.push_back(numreduce);
      this->compid.push_back(compid);
      numcompute++;
    }

    void report() {
      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);


      if(myid == printid) {
        CommBench::print_lib(this->lib);
        printf(" communication: ");
        {
          std::vector<std::vector<int>> matrix(numproc, std::vector<int>(numproc));
          size_t data = 0;
          for(int i = 0; i < this->numcomm; i++) {
            data += this->count[i] * sizeof(T);
            matrix[abs(this->recvid[i])][abs(this->sendid[i])]++;
          }
          CommBench::print_data(data);
          printf("\n");
          if(numproc < 64)
            for(int recv = 0; recv < numproc; recv++) {
              for(int send = 0; send < numproc; send++)
                if(matrix[recv][send])
                  printf("%d ", matrix[recv][send]);
                else
                  printf(". ");
              printf("\n");
            }
          printf("\n");
        }
        if(this->numcompute) {
          printf("computation: ");
          std::vector<int> input(numproc, 0);
          std::vector<int> output(numproc, 0);
          size_t inputdata = 0;
          size_t outputdata = 0;
          for(int i = 0; i < this->numcompute; i++) {
            inputdata += this->numreduce[i] * sizeof(T) * this->inputbuf[i].size();
            outputdata += this->numreduce[i] * sizeof(T);
            input[this->compid[i]] += this->inputbuf[i].size();
            output[this->compid[i]]++;
          }
          printf("input ");
          CommBench::print_data(inputdata);
          printf(" output ");
          CommBench::print_data(outputdata);
          printf("\n");
          if(numproc < 64)
            for(int p = 0; p < numproc; p++)
              if(output[p])
                printf("%d: %d -> %d\n", p, input[p], output[p]);
          printf("\n");
        }
      }
    }
  };

  template <typename T>
  void report_pipeline(std::vector<std::list<Coll<T>*>> &coll_batch) {

    int myid;
    int numproc;
    MPI_Comm_rank(comm_mpi, &myid);
    MPI_Comm_size(comm_mpi, &numproc);

    // REPORT PIPELINE
    if(myid == printid) {
      printf("********************************************\n\n");
      printf("pipeline depth %zu\n", coll_batch.size());
      printf("coll_list size %zu\n", coll_batch[0].size());
      printf("\n");
      int print_batch_size = (coll_batch.size() > 16 ? 16 : coll_batch.size());
      using Iter = typename std::list<ExaComm::Coll<T>*>::iterator;
      std::vector<Iter> coll_ptr(print_batch_size);
      for(int i = 0; i < print_batch_size; i++)
        coll_ptr[i] = coll_batch[i].begin();
      int collindex = 0;
      while(true) {
        bool finished = true;
        for(int i = 0; i < print_batch_size; i++)
          if(coll_ptr[i] != coll_batch[i].end())
            finished = false;
        if(finished)
          break;

        printf("proc %d index %d: |", myid, collindex);
        for(int i = 0; i < print_batch_size; i++)
          if(coll_ptr[i] != coll_batch[i].end()) {
            if((*coll_ptr[i])->numcomm)
              printf(" %d ", (*coll_ptr[i])->numcomm);
            else
              printf("   ");
            if((*coll_ptr[i])->numcomm + (*coll_ptr[i])->numcompute)
              CommBench::print_lib((*coll_ptr[i])->lib);
            else
              switch((*coll_ptr[i])->lib) {
                case CommBench::IPC  : printf(" I "); break;
                case CommBench::MPI  : printf(" M "); break;
                case CommBench::NCCL : printf(" N "); break;
                case CommBench::STAGE  : printf(" S "); break;
                case CommBench::numlib  : printf(" - "); break;
              }
            if((*coll_ptr[i])->numcompute)
              printf(" %d |", (*coll_ptr[i])->numcompute);
            else
              printf("   |");
            coll_ptr[i]++;
          }
          else
            printf("         |");
        printf("\n");
        collindex++;
      }
      printf("\n");
    }
  }

