
#if defined PORT_CUDA || defined PORT_HIP
  template <typename T>
  __global__ void reduce_kernel(T *output, size_t count, T **input, int numinput) {
     size_t i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i < count) {
       T acc = 0;
       for(int in = 0; in < numinput; in++)
         acc += input[in][i];
       output[i] = acc;
     }
  }
#else
  template <typename T>
  void reduce_kernel(T *output, size_t count, T **input, int numinput) {
    #pragma omp parallel for
    for(size_t i = 0; i < count; i++) {
      T acc = 0;
      for(int in = 0; in < numinput; in++)
        acc += inputbuf[in][i];
      output[i] = acc;
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
    std::vector<T**> inputbuf_d;
#ifdef PORT_CUDA
    std::vector<cudaStream_t*> stream;
#elif defined PORT_HIP
    std::vector<hipStream_t*> stream;
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
      if(myid == compid)
        for(int in = 0; in < inputbuf.size(); in++)
          MPI_Send(inputbuf.data() + in, sizeof(T*), MPI_BYTE, ROOT, 0, comm_mpi);
      if(printid == ROOT) {
        printf("add compute (%d) outputbuf %p, count %zu\n", compid, outputbuf, count);
        for(int in = 0; in < inputbuf.size(); in++) {
          T *inputbuf;
          MPI_Recv(&inputbuf, sizeof(T*), MPI_BYTE, compid, 0, comm_mpi, MPI_STATUS_IGNORE);
          printf("                 inputbuf %p\n", inputbuf);
        }
      }
      if(myid == compid) {
        this->inputbuf.push_back(inputbuf); // CPU COPY OF GPU POINTERS
        this->outputbuf.push_back(outputbuf);
        this->count.push_back(count);
        T **inputbuf_d;
#ifdef PORT_CUDA
        cudaMalloc(&inputbuf_d, inputbuf.size() * sizeof(T*));
        cudaMemcpy(inputbuf_d, inputbuf.data(), inputbuf.size() * sizeof(T*), cudaMemcpyHostToDevice);
        stream.push_back(new cudaStream_t);
        cudaStreamCreate(stream[numcomp]);
#elif defined PORT_HIP
        stream.push_back(new hipStream_t);
        hipStreamCreate(stream[numcomp]);
#endif
        this->inputbuf_d.push_back(inputbuf_d);
        numcomp++;
      }
    }

    void start() {
      int blocksize = 256;
      for(int comp = 0; comp < numcomp; comp++) {
#if defined PORT_CUDA || defined PORT_HIP
        reduce_kernel<T><<<(count[comp] + blocksize - 1) / blocksize, blocksize, 0, *stream[comp]>>> (outputbuf[comp], count[comp], inputbuf_d[comp], inputbuf[comp].size());
#else
        reduce_kernel (outputbuf[comp], count[comp], inputbuf_d[comp], inputbuf[comp].size());
#endif
      }
    }
    void wait() {
      for(int comp = 0; comp < numcomp; comp++) {
#ifdef PORT_CUDA
        cudaStreamSynchronize(*stream[comp]);
#elif defined PORT_HIP
        hipStreamSynchronize(*stream[comp]);
#endif
      }
    }

    void run() { start(); wait(); }

    void report() {
      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);
      std::vector<int> numcomp_all(numproc);
      std::vector<int> numinput_all(numproc);
      
      MPI_Allgather(&numcomp, 1, MPI_INT, numcomp_all.data(), 1, MPI_INT, MPI_COMM_WORLD);
      int numinput = 0;
      for(int comp = 0; comp < numcomp; comp++)
        numinput += inputbuf[comp].size();
      MPI_Allgather(&numinput, 1, MPI_INT, numinput_all.data(), 1, MPI_INT, MPI_COMM_WORLD);
      if(printid == ROOT) {
        printf("numcomp: ");
        for(int p = 0; p < numproc; p++)
          printf("%d(%d) ", numcomp_all[p], numinput_all[p]);
        printf("\n");
        printf("\n");
      }
    }

    void measure(int warmup, int numiter) {
      this->report();
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

      double data = 0;
      for(int comp = 0; comp < numcomp; comp++)
        data += count[comp] * sizeof(T) * (inputbuf[comp].size() + 1);
      MPI_Allreduce(MPI_IN_PLACE, &data, 1, MPI_DOUBLE, MPI_SUM, comm_mpi);

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
