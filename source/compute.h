
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
        acc += input[in][i];
      output[i] = acc;
    }
  }
#endif

  template <typename T>
  void allocate(T *&buffer, size_t n);

  template <typename T>
  class Compute {

    public:

    const MPI_Comm &comm_mpi = CommBench::comm_mpi;
    int numcomp = 0;

    std::vector<std::vector<T*>> inputbuf;
    std::vector<T*> outputbuf;
    std::vector<size_t> count;
    std::vector<T**> inputbuf_d;
#ifdef PORT_CUDA
    std::vector<cudaStream_t*> stream;
#elif defined PORT_HIP
    std::vector<hipStream_t*> stream;
#elif defined PORT_SYCL
    std::vector<sycl::queue*> queue;
#endif

    void add(std::vector<T*> &inputbuf, T *outputbuf, size_t count, int compid) {
      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);

      int printid = CommBench::printid;
      if(printid > -1 && printid < numproc) {
        if(myid == compid) {
          MPI_Send(&outputbuf, sizeof(T*), MPI_BYTE, printid, 0, comm_mpi);
          for(int in = 0; in < inputbuf.size(); in++)
            MPI_Send(inputbuf.data() + in, sizeof(T*), MPI_BYTE, printid, 0, comm_mpi);
        }
        if(myid == printid) {
          T *outputbuf;
          MPI_Recv(&outputbuf, sizeof(T*), MPI_BYTE, compid, 0, comm_mpi, MPI_STATUS_IGNORE);
          printf("add compute (%d) outputbuf %p, count %zu\n", compid, outputbuf, count);
          for(int in = 0; in < inputbuf.size(); in++) {
            T *inputbuf;
            MPI_Recv(&inputbuf, sizeof(T*), MPI_BYTE, compid, 0, comm_mpi, MPI_STATUS_IGNORE);
            printf("                 inputbuf %p\n", inputbuf);
          }
        }
      }
      if(myid == compid) {
        this->inputbuf.push_back(inputbuf); // CPU COPY OF GPU POINTERS
        this->outputbuf.push_back(outputbuf);
        this->count.push_back(count);
        T **inputbuf_d;
        allocate(inputbuf_d, inputbuf.size());
#ifdef PORT_CUDA
        cudaMemcpy(inputbuf_d, inputbuf.data(), inputbuf.size() * sizeof(T*), cudaMemcpyHostToDevice);
        stream.push_back(new cudaStream_t);
        cudaStreamCreate(stream[numcomp]);
#elif defined PORT_HIP
        hipMemcpy(inputbuf_d, inputbuf.data(), inputbuf.size() * sizeof(T*), hipMemcpyHostToDevice);
        stream.push_back(new hipStream_t);
        hipStreamCreate(stream[numcomp]);
#elif defined PORT_SYCL
	CommBench::q.memcpy(inputbuf_d, inputbuf.data(), inputbuf.size() * sizeof(T*)).wait();
        queue.push_back(new sycl::queue(sycl::gpu_selector_v));
#endif
        this->inputbuf_d.push_back(inputbuf_d);
        numcomp++;
      }
    }

    void start() {
      for(int comp = 0; comp < numcomp; comp++) {
#if defined PORT_CUDA || defined PORT_HIP
        int blocksize = 256;
        reduce_kernel<T><<<(count[comp] + blocksize - 1) / blocksize, blocksize, 0, *stream[comp]>>> (outputbuf[comp], count[comp], inputbuf_d[comp], inputbuf[comp].size());
#elif defined PORT_SYCL
        T *output = outputbuf[comp];
        int numinput = inputbuf[comp].size();
        T **input = inputbuf_d[comp];
        queue[comp]->parallel_for(sycl::range<1>{count[comp]}, [=] (sycl::id<1> i) {
          T acc = 0;
          for(int in = 0; in < numinput; in++)
            acc += input[in][i];
          output[i] = acc;
        });
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
#elif defined PORT_SYCL
        queue[comp]->wait();
#endif
      }
    }

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
      if(myid == printid) {
        printf("numcomp: ");
        for(int p = 0; p < numproc; p++)
          printf("%d(%d) ", numcomp_all[p], numinput_all[p]);
        printf("\n");
        printf("\n");
      }
    }

    void measure(int warmup, int numiter, size_t count) {
      int myid;
      int numproc;
      MPI_Comm_rank(comm_mpi, &myid);
      MPI_Comm_size(comm_mpi, &numproc);
      this->report();
      double times[numiter];
      if(myid == printid) {
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
        this->start();
        double start = MPI_Wtime() - time;
        this->wait();
        time = MPI_Wtime() - time;
        MPI_Allreduce(MPI_IN_PLACE, &start, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(iter < 0) {
          if(myid == printid)
            printf("startup %.2e warmup: %e\n", start, time);
        }
        else
          times[iter] = time;
      }
      std::sort(times, times + numiter,  [](const double & a, const double & b) -> bool {return a < b;});

      if(myid == printid) {
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
        size_t data = count * sizeof(T);
        printf("data: "); CommBench::print_data(data); printf("\n");
        printf("minTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", minTime * 1e6, minTime / data * 1e12, data / minTime / 1e9);
        printf("medTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", medTime * 1e6, medTime / data * 1e12, data / medTime / 1e9);
        printf("maxTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", maxTime * 1e6, maxTime / data * 1e12, data / maxTime / 1e9);
        printf("avgTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", avgTime * 1e6, avgTime / data * 1e12, data / avgTime / 1e9);
        printf("\n");
      }
    }
    void measure(int warmup, int numiter) {
      size_t count_total = 0;
      for(int comp = 0; comp < numcomp; comp++)
        count_total += count[comp] * (inputbuf[comp].size() + 1);
      MPI_Allreduce(MPI_IN_PLACE, &count_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_mpi);
      measure(warmup, numiter, count_total);
    }
  };

  // MEMORY MANAGEMENT

  template <typename T>
  void allocate(T *&buffer, size_t n) {
#ifdef PORT_CUDA
    cudaMalloc(&buffer, n * sizeof(T));
#elif defined PORT_HIP
    hipMalloc(&buffer, n * sizeof(T));
#elif defined PORT_SYCL
    buffer = sycl::malloc_device<T>(n, CommBench::q);
#else
    buffer = new T[n];
#endif
    buffsize += n;
  }

  template <typename T>
  void allocateHost(T *&buffer, size_t n) {
#ifdef PORT_CUDA
    cudaMallocHost(&buffer, n * sizeof(T));
#elif defined PORT_HIP
    hipHostMalloc(&buffer, n * sizeof(T));
#elif defined PORT_SYCL
    buffer = sycl::malloc_host<T>(n, CommBench::q);
#else
    buffer = new T[n];
#endif
  }

  template <typename T>
  void free(T *buffer) {
#ifdef PORT_CUDA
    cudaFree(buffer);
#elif defined PORT_HIP
    hipFree(buffer);
#elif defined PORT_SYCL
    sycl::free(buffer, CommBench::q);
#else
    delete[] buffer;
#endif
  }

  template <typename T>
  void freeHost(T *buffer) {
#ifdef PORT_CUDA
    cudaFreeHost(buffer);
#elif defined PORT_HIP
    hipHostFree(buffer);
#elif defined PORT_SYCL
    sycl::free(buffer, CommBench::q);
#else
    delete[] buffer;
#endif
  }


