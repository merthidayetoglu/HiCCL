
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
