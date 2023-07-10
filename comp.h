
#if defined PORT_CUDA || defined PORT_HIP
  template <typename T>
  __global__ void reduce_kernel(T* outputbuf, int numinput, T** inputbuf, size_t count) {
     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i < count) {
       T output = 0;
       for(int input = 0; input = numinput; input++)
         output += inputbuf[input][i];
       outputbuf[i] = output;
     }
  }
#else
  template <typename T>
    void reduce_kernel(T* outputbuf, int numinput, T** inputbuf, size_t count) {
    #pragma omp parallel for
    for(size_t i = 0; i < count; i++) {
      T output = 0;
      for(int input = 0; input < numinput; input++)
        output += inputbuf[input][i];
      outputbuf[i] = output;
    }
  }
#endif

  template <typename T>
  class Compute {

    MPI_Comm comm_mpi;
    int numcomp = 0;

    std::vector<std::vector<T*>> inputbuf;
    std::vector<T*> outputbuf;
    std::vector<size_t> count;
#ifdef PORT_CUDA
    std::vector<cudaStream_t*> stream;
#elif defined PORT_HIP
    std::vector<cudaStream_t*> stream;
#endif

    public:

    Compute(const MPI_Comm &comm_mpi_temp) {
      MPI_Comm_dup(comm_mpi_temp, &comm_mpi); // CREATE SEPARATE COMMUNICATOR EXPLICITLY
    }

    void add(std::vector<T*> &inputbuf, T *outputbuf, size_t count, int compid) {
      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);
      if(myid == compid) {
        this->inputbuf.push_back(inputbuf);
        this->outputbuf.push_back(outputbuf);
        this->count.push_back(count);
#ifdef PORT_CUDA
        stream.push_back(new cudaStream_t);
        cudaStreamCreate(stream[numcomp]);
#elif defined PORT_HIP
        stream.push_back(new hipStream_t);
        hipStreamCreate(stream[numcomp]);
#endif
        numcomp++;
      }
    }

    void start() {
      int blocksize = 256;
      for(int comp = 0; comp < numcomp; comp++) {
#if defined PORT_CUDA || PORT_HIP
        int numblocks = (count[comp] + blocksize - 1) / blocksize;
        reduce_kernel<<<blocksize, numblocks, 0, *stream[comp]>>> (outputbuf[comp], inputbuf[comp].size(), inputbuf[comp].data(), count[comp]);
#else
        reduce_kernel(outputbuf[comp], inputbuf[comp].size(), inputbuf[comp].data(), count[comp]);
#endif
      }
    }
    void wait() {
      for(int comp = 0; comp < numcomp; comp++)
#ifdef PORT_CUDA
        cudaStreamSynchronize(*stream[comp]);
#elif defined PORT_HIP
        hipStreamSynchronize(*stream[comp]);
#endif
    }

    void run() { start(); wait(); }

    void report() {
      if(printid == ROOT) {
        printf("numcomp %d\n", numcomp);
        for(int comp = 0; comp < numcomp; comp++)
          printf("comp %d count %zu\n", comp, count[comp]);
      }
    }

    void measure(int warmup, int numiter) {
      double times[numiter];
      if(printid == ROOT) {
        printf("Measure Reduction Kernel\n");
        printf("%d warmup iterations (in order)\n", warmup);
      }
      for (int iter = -warmup; iter < numiter; iter++) {
#ifdef PORT_CUDA
        cudaDeviceSynchronize();
#elif defined PORT_HIP
        hipDeviceSynchronize();
#endif
        MPI_Barrier(MPI_COMM_WORLD);
        double time = MPI_Wtime();
        run();
        time = MPI_Wtime() - time;

        MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(iter < 0) {
          if(printid == ROOT)
            printf("warmup: %e\n", time);
        }
        else
          times[iter] = time;
      }
      std::sort(times, times + numiter,  [](const double & a, const double & b) -> bool {return a < b;});

      if(printid == ROOT) {
        printf("%d measurement iterations (sorted):\n", numiter);
        for(int iter = 0; iter < numiter; iter++) {
          printf("time: %.4e", times[iter]);
          if(iter == 0)
            printf(" -> min\n");
          else if(iter == numiter / 2)
            printf(" -> median\n");
          else if(iter == numiter - 1)
            printf(" -> max\n");
          else
            printf("\n");
        }
        printf("\n");
        double minTime = times[0];
        double medTime = times[numiter / 2];
        double maxTime = times[numiter - 1];
        double avgTime = 0;
        for(int iter = 0; iter < numiter; iter++)
          avgTime += times[iter];
        avgTime /= numiter;
        double data = accumulate(count.begin(), count.end(), 0) * sizeof(T);
        if (data < 1e3)
          printf("data: %d bytes\n", (int)data);
        else if (data < 1e6)
          printf("data: %.4f KB\n", data / 1e3);
        else if (data < 1e9)
          printf("data: %.4f MB\n", data / 1e6);
        else if (data < 1e12)
          printf("data: %.4f GB\n", data / 1e9);
        else
          printf("data: %.4f TB\n", data / 1e12);
        printf("minTime: %.4e us, %.4e s/GB, %.4e GB/s\n", minTime * 1e6, minTime / data * 1e9, data / minTime / 1e9);
        printf("medTime: %.4e us, %.4e s/GB, %.4e GB/s\n", medTime * 1e6, medTime / data * 1e9, data / medTime / 1e9);
        printf("maxTime: %.4e us, %.4e s/GB, %.4e GB/s\n", maxTime * 1e6, maxTime / data * 1e9, data / maxTime / 1e9);
        printf("avgTime: %.4e us, %.4e s/GB, %.4e GB/s\n", avgTime * 1e6, avgTime / data * 1e9, data / avgTime / 1e9);
        printf("\n");
      }
    }
  };
