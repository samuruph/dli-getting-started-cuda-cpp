#include "dli.h"

__global__ void grid_kernel(dli::temperature_grid_f in, float *out)
{
  int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
  int number_of_threads = blockDim.x * gridDim.x;

  for (int id = thread_index; id < in.size(); id += number_of_threads) 
  {
    out[id] = dli::compute(id, in);
  }
}

int ceil_div(int a, int b) 
{
  return (a + b - 1) / b;
}

void simulate(dli::temperature_grid_f temp_in, float *temp_out, cudaStream_t stream)
{
  int block_size = 1024;
  int grid_size = ceil_div(temp_in.size(), block_size);

  grid_kernel<<<grid_size, block_size, 0, stream>>>(temp_in, temp_out);
}
