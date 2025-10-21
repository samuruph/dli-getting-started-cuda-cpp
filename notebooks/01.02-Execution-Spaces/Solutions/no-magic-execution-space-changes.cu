#include "dli.h"

int main() {
  dli::where_am_I("CPU");

  thrust::universal_vector<int> vec{1};
  thrust::for_each(thrust::device, vec.begin(), vec.end(),
                   [] __host__ __device__(int) { dli::where_am_I("GPU"); });

  thrust::for_each(thrust::host, vec.begin(), vec.end(),
                   [] __host__ __device__(int) { dli::where_am_I("CPU"); });

  dli::where_am_I("CPU");
}