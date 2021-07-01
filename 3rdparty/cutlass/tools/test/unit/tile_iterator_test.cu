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

#include "cutlass_unit_test.h"
#include "cutlass/shape.h"
#include "cutlass/tile_iterator.h"
#include "gtest/gtest.h"

using ::cutlass::Coord;
using ::cutlass::Fragment;
using ::cutlass::IteratorAdvance;
using ::cutlass::make_Coord;
using ::cutlass::MemorySpace;
using ::cutlass::Shape;
using ::cutlass::TileLoadIterator;
using ::cutlass::TileTraits;
using ::testing::Test;


// Returns randomly initialized array
//
// Caller is responsible for deallocation.
float* malloc_randomly_initialized_array(int elements) {
  float* matrix = (float*)calloc(sizeof(float), elements);
  for (int i = 0; i < elements; i++) {
    matrix[i] = float((rand() - RAND_MAX/2) % 10);
  }
  return matrix;
}

#define kWarpSize 32
#define kCtaWarpCnt 6
#define kDimXPerWarp 16
#define kDimYPerWarp 2
#define kWarpTileWidth kDimXPerWarp
#define kDimYPerThread (kWarpSize / kDimYPerWarp)
#define kDimX 2400
#define kDimY 800

struct TileThreadOffset {
public:
  TileThreadOffset() : xidx(0), yidx(0) {}
  TileThreadOffset(int x, int y) : xidx(x), yidx(y) {}
  
  __host__ __device__ Coord<4> operator()() const {
    int column = (yidx / kDimYPerWarp) * kDimXPerWarp +
                 (yidx & (kDimYPerWarp - 1)) * kDimYPerThread;
    return make_Coord(0, column, xidx, 0);
  }
  
 private:
  int xidx, yidx;
};


TEST(TileIteratorTest, BasicCpuSideIterateTile) {
  // Basic test demonstrating CPU-side tile iteration mimicking a 16x16 tile load/warp with 6 warp
  // CTAs iterating over the Y.
  
  float* matrix = malloc_randomly_initialized_array(kDimX*kDimY);

  typedef Shape</*kD=*/1, /*kH=*/kCtaWarpCnt * kDimXPerWarp, /*kW=*/kDimXPerWarp> TileShape;
  typedef TileLoadIterator<
   TileTraits<TileShape,
              /* Delta = */ Shape</*kD=*/1, /*kH=*/1, /*kW=*/1>,
              /* Iter = */ Shape</*kD=*/1, /*kH=*/kDimYPerThread, /*kW=*/1>,
              TileThreadOffset, /*AccessSize=*/1>,
    float, IteratorAdvance::kH, MemorySpace::kGlobal> GlobalTileLoader;
  typedef GlobalTileLoader::Fragment BufferType;
  // Iterate: gridDim(1, 1, kDimX / kDimXPerWarp), blockDim(1, kDimXPerWarp, kDimYPerWarp)
  for (int blockIdx_x = 0; blockIdx_x < kDimX / kDimXPerWarp; blockIdx_x++) {
    for (int threadIdx_x = 0; threadIdx_x < kDimXPerWarp; threadIdx_x++) {
      for (int threadIdx_y = 0; threadIdx_y < kCtaWarpCnt * kDimYPerWarp; threadIdx_y++) {
        GlobalTileLoader loader(
            GlobalTileLoader::Params(matrix,
                                     /* stride_d=*/1, /*stride_h=*/kDimX, /*stride_w=*/1),
            make_Coord(/*d=*/0, /*h=*/0, /*w=*/blockIdx_x * kDimXPerWarp),
            TileThreadOffset(threadIdx_x, threadIdx_y));
        BufferType b;
        for (int yidx = 0; (yidx + threadIdx_y * kWarpTileWidth) < kDimY;
             yidx += kCtaWarpCnt*kWarpTileWidth) {
          
          loader.load_post_increment(b);
          for (int i = 0; i < BufferType::kElements; i++) {
            int matrix_idx = blockIdx_x * kDimXPerWarp + threadIdx_x + // row offset
                             kDimX * ((threadIdx_y & (kDimYPerWarp - 1)) * kDimYPerThread +
                                      (threadIdx_y / kDimYPerWarp) * kWarpTileWidth + i + yidx);
            ASSERT_EQ(b[i], matrix[matrix_idx])
                << "blockIdx.x = " << blockIdx_x << " threadIdx.x = " << threadIdx_x
                << " threadIdx.y = " << threadIdx_y << " yidx = " << yidx
                << " tile_idx = " << i << " matrix_idx = " << matrix_idx;
          }
        }
      }
    }
  }
  free(matrix);
}
