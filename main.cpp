#define PORT_CUDA
#include "hiccl.h"

using namespace CommBench;

int main() {

  size_t numbytes = 1e9;
  char *sendbuf;
  char *recvbuf;
  allocate(sendbuf, numbytes);
  allocate(recvbuf, numbytes);

  HiCCL::Comm<char> coll;

  free(sendbuf);
  free(recvbuf);

  return 0;
}
