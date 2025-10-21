#pragma once

#include <algorithm>
#include <vector>
#include <fstream>

#include <cuda/std/mdspan>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace dli {

// Constants
constexpr float C0 = 299792458.0f;            // Speed of light [metres per second]
constexpr float epsilon_0 = 8.854187817e-12f; // Permittivity of free space
constexpr float mu_0 = 4 * M_PI * 1e-7f;      // Permeability of free space

constexpr int steps = 10;

constexpr int tile_size = 16;
constexpr int block_threads = tile_size * tile_size;

using temperature_grid_f =
    cuda::std::mdspan<float, cuda::std::dextents<int, 2>>;

}
