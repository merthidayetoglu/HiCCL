template <typename T, typename Comm>
void measure(size_t count, int warmup, int numiter, Comm &comm) {

  int myid;
  int numproc;
  MPI_Comm_rank(comm_mpi, &myid);
  MPI_Comm_size(comm_mpi, &numproc);

  double times[numiter];
  if(myid == printid)
    printf("%d warmup iterations (in order):\n", warmup);
  for (int iter = -warmup; iter < numiter; iter++) {

#ifdef PORT_CUDA
    cudaDeviceSynchronize();
#elif defined PORT_HIP
    hipDeviceSynchronize();
#endif
    MPI_Barrier(comm_mpi);
    double time = MPI_Wtime();

    comm.run();

    time = MPI_Wtime() - time;

    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, comm_mpi);
    if(iter < 0) {
      if(myid == printid)
        printf("warmup: %e\n", time);
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
    double data = count * sizeof(T);
    printf("Total data: "); CommBench::print_data(data); printf("\n");
    printf("Total minTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", minTime * 1e6, minTime / data * 1e12, data / minTime / 1e9);
    printf("Total medTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", medTime * 1e6, medTime / data * 1e12, data / medTime / 1e9);
    printf("Total maxTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", maxTime * 1e6, maxTime / data * 1e12, data / maxTime / 1e9);
    printf("Total avgTime: %.4e us, %.4e ms/GB, %.4e GB/s\n", avgTime * 1e6, avgTime / data * 1e12, data / avgTime / 1e9);
    printf("\n");
  }
}

template <typename T, typename Comm>
void validate(T *sendbuf_d, T *recvbuf_d, size_t count, int patternid, int root, Comm &comm) {

  int myid;
  int numproc;
  MPI_Comm_rank(comm_mpi, &myid);
  MPI_Comm_size(comm_mpi, &numproc);

  enum pattern {dummy, gather, scatter, broadcast, reduce, alltoall, allgather, reducescatter, allreduce};

  T *sendbuf;
  T *recvbuf;
#ifdef PORT_CUDA
  cudaMallocHost(&sendbuf, count * numproc * sizeof(T));
  cudaMallocHost(&recvbuf, count * numproc * sizeof(T));
  cudaMemset(recvbuf_d, -1, count * numproc * sizeof(T));
#elif defined PORT_HIP
  hipHostMalloc(&sendbuf, count * numproc * sizeof(T));
  hipHostMalloc(&recvbuf, count * numproc * sizeof(T));
  hipMemset(recvbuf_d, -1, count * numproc * sizeof(T));
#elif defined PORT_SYCL
  sendbuf = sycl::malloc_host<T>(count * numproc, CommBench::q);
  recvbuf = sycl::malloc_host<T>(count * numproc, CommBench::q);
  CommBench::q.memset(recvbuf_d, -1, count); // call a kernel;
#endif
  #pragma omp parallel for
  for(size_t i = 0; i < count * numproc; i++)
    sendbuf[i] = i;
#ifdef PORT_CUDA
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaMemcpyAsync(sendbuf_d, sendbuf, count * numproc * sizeof(T), cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
#elif defined PORT_HIP
  hipStream_t stream;
  hipStreamCreate(&stream);
  hipMemcpyAsync(sendbuf_d, sendbuf, count * numproc * sizeof(T), hipMemcpyHostToDevice, stream);
  hipStreamSynchronize(stream);
#elif defined PORT_SYCL
  CommBench::q.memcpy(sendbuf_d, sendbuf, count * numproc * sizeof(T));
  CommBench::q.wait();
#endif
  MPI_Barrier(comm_mpi);

  comm.run();

#ifdef PORT_CUDA
  cudaMemcpyAsync(recvbuf, recvbuf_d, count * numproc * sizeof(T), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
#elif defined PORT_HIP
  hipMemcpyAsync(recvbuf, recvbuf_d, count * numproc * sizeof(T), hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);
#elif defined PORT_SYCL
  CommBench::q.memcpy(recvbuf, recvbuf_d, count * numproc * sizeof(T));
  CommBench::q.wait();
#endif

  unsigned long errorcount = 0;
  bool pass = true;
  switch(patternid) {
    case gather: if(myid == printid) printf("VERIFY GATHER ROOT = %d: ", root);
      if(myid == root)
        for(int p = 0; p < numproc; p++)
          for(size_t i = 0; i < count; i++) {
            // printf("myid %d recvbuf[%zu] = %d\n", myid, p * count + i, recvbuf[p * count + i]);
            if(recvbuf[p * count + i] != i) {
              pass = false;
              errorcount++;
            }
          }
      break;
    case scatter: if(myid == printid) printf("VERIFY SCATTER ROOT = %d: ", root);
      for(size_t i = 0; i < count; i++) {
        // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
        if(recvbuf[i] != myid * count + i) {
          pass = false;
          errorcount++;
        }
      }
      break;
    case broadcast: if(myid == printid) printf("VERIFY BCAST ROOT = %d: ", root);
      for(size_t i = 0; i < count * numproc; i++) {
        // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
        if(recvbuf[i] != i) {
          pass = false;
          errorcount++;
        }
      }
      break;
    case reduce: if(myid == printid) printf("VERIFY REDUCE ROOT = %d: ", root);
      if(myid == root)
        for(size_t i = 0; i < count * numproc; i++) {
          // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
          if(recvbuf[i] != i * numproc) {
            pass = false;
            errorcount++;
          }
        }
      break;
    case alltoall: if(myid == printid) printf("VERIFY ALL-TO-ALL: ");
      for(int p = 0; p < numproc; p++)
        for(size_t i = 0; i < count; i++) {
          // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
          if(recvbuf[p * count + i] != myid * count + i) {
            pass = false;
            errorcount++;
          }
        }
      break;
    case allgather: if(myid == printid) printf("VERIFY ALL-GATHER: ");
      for(int p = 0; p < numproc; p++)
        for(size_t i = 0; i < count; i++) {
          // if(myid == printid) printf("myid %d recvbuf[%d] = %d (%d)\n", myid, p * count + i, recvbuf[p * count + i], i);
          if(recvbuf[p * count + i] != i) {
            pass = false;
            errorcount++;
          }
        }
      break;
    case reducescatter: if(myid == printid) printf("VERIFY REDUCE-SCATTER: ");
      for(size_t i = 0; i < count; i++) {
        // if(myid == printid) printf("myid %d recvbuf[%d] = %d (%d)\n", myid, i, recvbuf[i], (myid * count + i) * numproc);
        if(recvbuf[i] != (myid * count + i) * numproc) {
          pass = false;
          errorcount++;
        }
      }
      break;
    case allreduce: if(myid == printid) printf("VERIFY ALL-REDUCE: ");
      for(size_t i = 0; i < count * numproc; i++) {
        // if(myid == printid) printf("myid %d recvbuf[%d] = %d (%d)\n", myid, i, recvbuf[i], i * numproc);
        if(recvbuf[i] != i * numproc) {
          pass = false;
          errorcount++;
        }
      }
      break;
    default:
      pass = false;
      break;
  }
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_C_BOOL, MPI_LAND, comm_mpi);
  if(myid == printid) {
    if(pass)
      printf("PASSED!\n");
    else
      printf("FAILED!!!\n");
  }
  if(!pass) {
    std::vector<unsigned long> errorcounts(numproc);
    MPI_Allgather(&errorcount, 1, MPI_UNSIGNED_LONG, errorcounts.data(), 1, MPI_UNSIGNED_LONG, comm_mpi);
    MPI_Allreduce(MPI_IN_PLACE, &errorcount, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm_mpi);
    if(myid == printid) {
      printf("count %zu total errorcount %zu\n", count, errorcount);
      for(int proc = 0; proc < numproc; proc++)
        printf("proc %d errorcount:  %zu\n", proc, errorcounts[proc]);
    }
  }

#ifdef PORT_CUDA
  cudaFreeHost(sendbuf);
  cudaFreeHost(recvbuf);
#elif defined PORT_HIP
  hipHostFree(sendbuf);
  hipHostFree(recvbuf);
#endif
}

