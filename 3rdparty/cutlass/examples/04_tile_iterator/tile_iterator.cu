/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to use the TileIterator in CUTLASS to load data from addressable
  memory, and store it back into addressable memory.

  TileIterator is a core concept in CUTLASS that enables efficient loading and storing of data from
  and to addressable memory. The TileIterator accepts a TileTraits type, which defines the shape of a 
  tile and the distribution of accesses by individual entities, either threads or others.

  In this example, a LoadTileIterator is used to load elements from a tile in global memory, stored in 
  column-major layout, into a fragment, and a corresponding StoreTileIterator is used to store the
  elements back into global memory (in the same column-major layout).

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  This example uses CUTLASS utilities to ease the matrix operations.
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// CUTLASS includes
#include "cutlass/tile_iterator.h"
#include "cutlass/tile_traits_standard.h"

//
// CUTLASS utility includes
//

// Defines operator<<() to write TensorView objects to std::ostream
#include "tools/util/tensor_view_io.h"

// Defines cutlass::HostMatrix<>
#include "tools/util/host_matrix.h"

// Defines cutlass::reference::device::TensorInitialize()
#include "tools/util/reference/device/tensor_elementwise.h"

// Defines cutlass::reference::host::TensorEquals()
#include "tools/util/reference/host/tensor_elementwise.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines load and store tile iterators to load and store a M-by-K tile, in
// column-major layout, from and back into global memory.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Traits>
__global__ void cutlass_tile_iterator_load_store_global(
  float const *input,
  float *output,
  int M,
  int K) {

    // Define a tile load iterator
    typedef cutlass::TileLoadIterator<
        Traits,                         // the Traits type, defines shape/distribution of accesses
        float,                          // elements are of type float
        cutlass::IteratorAdvance::kH,   // post-increment accesses advance in strided (as opposed to
                                        //     contiguous dimension 
        cutlass::MemorySpace::kGlobal   // iterator loads from global memory 
        > TileLoadIterator;

    // Defines a tile store iterator
    typedef cutlass::TileStoreIterator<
        Traits,                         // the Traits type, defines shape/distribution of accesses
        float,                          // elements are of type float
        cutlass::IteratorAdvance::kH,   // post-increment accesses advance in strided (as opposed to
                                        //     contiguous) dimension
        cutlass::MemorySpace::kGlobal   // iterator stores into global memory
        > TileStoreIterator;

    // Defines a predicate vector for managing statically sized vector of boolean predicates
    typedef typename TileLoadIterator::PredicateVector PredicateVector;

    // The parameters specified to the iterators. These include the pointer to the source of
    // addressable memory, and the strides and increments for each of the tile's dimensions  
    typename TileLoadIterator::Params load_params;
    typename TileStoreIterator::Params store_params;

    // Initializing the parameters for both of the iterators. The TileLoadIterator accesses the
    // input matrix and TileStoreIterator accesses the output matrix. The strides are set
    // identically since the data is being stored in the same way as it is loaded (column-major
    // mapping).
    load_params.initialize(input, M*K, M, 1);
    store_params.initialize(output, M*K, M, 1);
   
    // Constructing the tile load and store iterators, and the predicates vector
    TileLoadIterator load_iterator(load_params);
    TileStoreIterator store_iterator(store_params);
    PredicateVector predicates;

    // Initializing the predicates with bounds set to <1, K, M>. This protects out-of-bounds loads.
    load_iterator.initialize_predicates(predicates.begin(), cutlass::make_Coord(1, K, M));

    // The fragment in which the elements are loaded into and stored from.
    typename TileLoadIterator::Fragment fragment;

    // Loading a tile into a fragment and advancing to the next tile's position
    load_iterator.load_post_increment(fragment, predicates.begin());
    // Storing a tile from fragment and advancing to the next tile's position
    store_iterator.store_post_increment(fragment);
}


///////////////////////////////////////////////////////////////////////////////////////////////////

// Launches cutlass_tile_iterator_load_store_global kernel
cudaError_t test_cutlass_tile_iterator() {
  cudaError_t result = cudaSuccess;

  // Creating a M-by-K (128-by-8) tile for this example.
  static int const M = 128;
  static int const K = 8;
  // The kernel is launched with 128 threads per thread block.
  static int const kThreadsPerThreadBlock = 128;
  // Define the tile type
  typedef cutlass::Shape<1, 8, 128> Tile;

  // CUTLASS provides a standard TileTraits type, which chooses the 'best' shape to enable warp 
  // raking along the contiguous dimension if possible.
  typedef cutlass::TileTraitsStandard<Tile, kThreadsPerThreadBlock> Traits;

  // M-by-K input matrix of float
  cutlass::HostMatrix<float> input(cutlass::MatrixCoord(M, K));

  // M-by-K output matrix of float
  cutlass::HostMatrix<float> output(cutlass::MatrixCoord(M, K));

  //
  // Initialize input matrix with linear combination.
  //

  cutlass::Distribution dist;

  // Linear distribution in column-major format.
  dist.set_linear(1, 1, M);

  // Arbitrary RNG seed value. Hard-coded for deterministic results.
  int seed = 2080;

  cutlass::reference::device::TensorInitialize(
    input.device_view(),                                // concept: TensorView
    seed,
    dist);

  // Initialize output matrix to all zeroes.
  output.fill(0);

  // Launch kernel to load and store tiles from/to global memory.
  cutlass_tile_iterator_load_store_global<Traits><<<
      dim3(1, 1, 1),
      dim3(kThreadsPerThreadBlock, 1)
    >>>(input.device_data(), output.device_data(), M, K);

  result = cudaDeviceSynchronize();

  if (result != cudaSuccess) {
    return result;
  }

  // Copy results to host
  output.sync_host();

  // Verify results
  for(int i = 0; i < M; ++i) {
    for(int j = 0; j < K; ++j) {
      if(output.at(cutlass::make_Coord(i, j)) != float(M*j+i+1)){
        std::cout << "FAILED: (" << i << ", " << j
                  << ") -- expected: " << (M*j+i+1)
                  << ", actual: " << output.at(cutlass::make_Coord(i, j))
                  << std::endl;
        result = cudaErrorUnknown;
        break;
      }
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to tile_iterator example.
//
// usage:
//
//   04_tile_iterator
//
int main(int argc, const char *arg[]) {
  
  // Properties of CUDA device
  cudaDeviceProp device_properties;
    
  // Assumne the device id is 0.
  int device_id = 0;

  cudaError_t result = cudaGetDeviceProperties(&device_properties, device_id);
  if (result != cudaSuccess) {
    std::cerr << "Failed to get device properties: " 
      << cudaGetErrorString(result) << std::endl;
    return -1;
  }


  //
  // Run the CUTLASS tile iterator test.
  //

  result = test_cutlass_tile_iterator();

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

