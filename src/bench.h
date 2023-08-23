template <typename T, typename Comm>
void measure(size_t count, int warmup, int numiter, Comm &comm) {

  int myid;
  int numproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  int numthread = -1;
  #pragma omp parallel
  #pragma omp master
  numthread = omp_get_num_threads();

  double times[numiter];
  if(myid == ROOT)
    printf("%d warmup iterations (in order) numthread %d:\n", warmup, numthread);
  for (int iter = -warmup; iter < numiter; iter++) {

#ifdef PORT_CUDA
    cudaDeviceSynchronize();
#elif defined PORT_HIP
    hipDeviceSynchronize();
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();

    comm.run();

    time = MPI_Wtime() - time;

    MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(iter < 0) {
      if(myid == ROOT)
        printf("warmup: %e\n", time);
    }
    else
      times[iter] = time;
  }
  std::sort(times, times + numiter,  [](const double & a, const double & b) -> bool {return a < b;});

  if(myid == ROOT) {
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
    printf("Total minTime: %.4e us, %.4e s/GB, %.4e GB/s\n", minTime * 1e6, minTime / data * 1e9, data / minTime / 1e9);
    printf("Total medTime: %.4e us, %.4e s/GB, %.4e GB/s\n", medTime * 1e6, medTime / data * 1e9, data / medTime / 1e9);
    printf("Total maxTime: %.4e us, %.4e s/GB, %.4e GB/s\n", maxTime * 1e6, maxTime / data * 1e9, data / maxTime / 1e9);
    printf("Total avgTime: %.4e us, %.4e s/GB, %.4e GB/s\n", avgTime * 1e6, avgTime / data * 1e9, data / avgTime / 1e9);
    printf("\n");
  }
}

template <typename T, typename Comm>
void validate(T *sendbuf_d, T *recvbuf_d, size_t count, int patternid, Comm &comm) {

  int myid;
  int numproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &numproc);

  enum pattern {dummy, scatter, gather, broadcast, reduce, alltoall, allgather, reducescatter, allreduce};

  T *recvbuf;
  T *sendbuf;
#ifdef PORT_CUDA
  cudaMallocHost(&sendbuf, count * numproc * sizeof(T));
  cudaMallocHost(&recvbuf, count * numproc * sizeof(T));
  cudaMemset(recvbuf_d, -1, count * numproc * sizeof(T));
#elif defined PORT_HIP
  hipHostMalloc(&sendbuf, count * numproc * sizeof(T));
  hipHostMalloc(&recvbuf, count * numproc * sizeof(T));
  hipMemset(recvbuf_d, -1, count * numproc * sizeof(T));
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
#endif
  MPI_Barrier(MPI_COMM_WORLD);

  comm.run();

#ifdef PORT_CUDA
  cudaMemcpyAsync(recvbuf, recvbuf_d, count * numproc * sizeof(T), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
#elif defined PORT_HIP
  hipMemcpyAsync(recvbuf, recvbuf_d, count * numproc * sizeof(T), hipMemcpyDeviceToHost, stream);
  hipStreamSynchronize(stream);
#endif

  unsigned long errorcount = 0;
  bool pass = true;
  switch(patternid) {
    case scatter: if(myid == ROOT) printf("VERIFY SCATTER ROOT = %d: ", ROOT);
      for(size_t i = 0; i < count; i++) {
        // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
        if(recvbuf[i] != myid * count + i)
          pass = false;
      }
      break;
    case gather: if(myid == ROOT) printf("VERIFY GATHER ROOT = %d: ", ROOT);
      if(myid == ROOT)
        for(int p = 0; p < numproc; p++)
          for(size_t i = 0; i < count; i++) {
            // printf("myid %d recvbuf[%zu] = %d\n", myid, p * count + i, recvbuf[p * count + i]);
            if(recvbuf[p * count + i] != i)
              pass = false;
          }
      break;
    case broadcast: if(myid == ROOT) printf("VERIFY BCAST ROOT = %d: ", ROOT);
      for(size_t i = 0; i < count * numproc; i++) {
        // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
        if(recvbuf[i] != i)
          pass = false;
      }
      break;
    case reduce: if(myid == ROOT) printf("VERIFY REDUCE ROOT = %d: ", ROOT);
      if(myid == ROOT)
        for(size_t i = 0; i < count * numproc; i++) {
          // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
          if(recvbuf[i] != i * numproc) {
            pass = false;
            errorcount++;
          }
        }
      break;
    case alltoall: if(myid == ROOT) printf("VERIFY ALL-TO-ALL: ");
      for(int p = 0; p < numproc; p++)
        for(size_t i = 0; i < count; i++) {
          // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
          if(recvbuf[p * count + i] != myid * count + i)
            pass = false;
        }
      break;
    case allgather: if(myid == ROOT) printf("VERIFY ALL-GATHER: ");
      for(int p = 0; p < numproc; p++)
        for(size_t i = 0; i < count; i++) {
          // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
          if(recvbuf[p * count + i] != i)
            pass = false;
        }
      break;
    case reducescatter: if(myid == ROOT) printf("VERIFY REDUCE-SCATTER: ");
      for(size_t i = 0; i < count; i++) {
        // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
        if(recvbuf[i] != (myid * count + i) * numproc)
          pass = false;
      }
      break;
    case allreduce: if(myid == ROOT) printf("VERIFY ALL-REDUCE: ");
      for(size_t i = 0; i < count * numproc; i++) {
        // printf("myid %d recvbuf[%d] = %d\n", myid, i, recvbuf[i]);
        if(recvbuf[i] != i * numproc)
          pass = false;
      }
      break;
    default:
      pass = false;
      break;
  }
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
  if(myid == ROOT) {
    if(pass)
      printf("PASSED!\n");
    else
      printf("FAILED!!!\n");
  }
  if(!pass) {
    std::vector<unsigned long> errorcounts(numproc);
    MPI_Allgather(&errorcount, 1, MPI_UNSIGNED_LONG, errorcounts.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &errorcount, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(myid == ROOT) {
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

