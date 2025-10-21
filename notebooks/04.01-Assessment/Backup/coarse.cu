#include "dli.h"

__global__ void kernel(dli::temperature_grid_f fine,
                       dli::temperature_grid_f coarse) {
  int coarse_row = blockIdx.x / coarse.extent(1);
  int coarse_col = blockIdx.x % coarse.extent(1);
  int row = threadIdx.x / dli::tile_size;
  int col = threadIdx.x % dli::tile_size;
  int fine_row = coarse_row * dli::tile_size + row;
  int fine_col = coarse_col * dli::tile_size + col;

  float thread_value = fine(fine_row, fine_col);

  // FIXME(Step 3):
  // Compute the sum of `thread_value` across threads of a thread block
  // using `cub::BlockReduce`
  float block_sum = ...;

  // FIXME(Step 3):
  // `cub::BlockReduce` returns block sum in thread 0, make sure to write
  // result only from the first thread of the block
  coarse(coarse_row, coarse_col) = block_average;
}

// Don't change the signature of this function
void coarse(dli::temperature_grid_f fine, dli::temperature_grid_f coarse) {
  kernel<<<coarse.size(), dli::block_threads>>>(fine, coarse);
}
