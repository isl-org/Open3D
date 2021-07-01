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
/*! \file
    \brief Defies functors for mapping blockIdx to partitions of the GEMM computation.
*/
#pragma once

#include "cutlass/coord.h"
#include "cutlass/gemm/gemm_coord.h"

namespace cutlass {
namespace gemm {

struct swizzleDirection {
  enum Kind { Boustrophedon, OneDirection };
};
// helper template function
template <enum swizzleDirection::Kind>
CUTLASS_DEVICE int getLinearIdx(int groups) {
  // groupCols is not needed for OneDirection Swizzle
  return blockIdx.y * gridDim.x + blockIdx.x;
}
template <>
CUTLASS_DEVICE int getLinearIdx<swizzleDirection::Boustrophedon>(int groups) {
  // reverse blockIdx.x for some columns
  if ((blockIdx.y / groups) % 2 == 1)
    return blockIdx.y * gridDim.x + (gridDim.x - blockIdx.x - 1);
  else
    return blockIdx.y * gridDim.x + blockIdx.x;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

/*!@defgroup IdentityBlockSwizzle Identity Block Swizzle
@{
    Block Swizzle provides the mapping logic between a block in the physical memory of Matrix C and
Thread Block
    Identiy Block Swizzle effective maps blocks in leading dimension order (column major) with
thread block
    in leading dimension order (blockIdx.x)
    blockIdx.z is mapped with batch_count for batched GEMM
@}
*/
struct IdentityBlockSwizzle {
  /// Ctor. aka ColumnMajorBlockSwizzle<1>
  CUTLASS_HOST_DEVICE IdentityBlockSwizzle() {}

  /// Swizzle the block index.
  CUTLASS_DEVICE dim3 swizzle() { return dim3(blockIdx.x, blockIdx.y, blockIdx.z); }

  ///
  CUTLASS_HOST_DEVICE dim3 get_grid_layout(GemmCoord const &problem_size,
                                           Coord<3> const &OutputTile) {
    /*OutputTile and problem_size are both in KNM order*/
    dim3 grid;
    grid.x = (problem_size.m() + OutputTile[2] - 1) / OutputTile[2];
    grid.y = (problem_size.n() + OutputTile[1] - 1) / OutputTile[1];
    grid.z = problem_size.batch();
    return grid;
  }

  ///get threadblock offset, without considering tha batch dim
  CUTLASS_DEVICE Coord<3> get_threadblock_offset(Coord<3> const &OutputTile) {
    dim3 block = swizzle();
    Coord<3> threadblock_offset =
        make_Coord(0, block.y * OutputTile[1], block.x * OutputTile[2]);
    return threadblock_offset;
  }

  ///
  CUTLASS_DEVICE int get_batch_id() {
    dim3 block = swizzle();
    return block.z;
  }

  /// check if at the last partition
  CUTLASS_DEVICE bool is_last_partition() {
    if (get_batch_id() == (gridDim.z - 1))
      return true;
    else
      return false;
  }

  ///
  CUTLASS_DEVICE Coord<3> get_threadblock_bounds(GemmCoord const &problem_size,
                                                 int partitionK_range) {
    // every partition except the last one has a smaller range
    // partitionK_range is the bounds for every partition except the last one
    // the last partition's bounds is the same with problem size
    if(is_last_partition())
      return problem_size.knm();
    else
      return make_Coord(partitionK_range, problem_size.n(), problem_size.m());
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
ColumnMajorBlockSwizzle<1, OneDirection> is equivalent with IdentityBlockSwizzle
groupCols has the effect of controlling the schedulling of thread blocks
settings with different groupCols can contribute to the overall performance by affecting L2 cache
hit rate

consider a regular thread block mapping btween matrix C and different thread blocks
note that C is column major, and the leading dimension of thread block id is blockIdx.x

let's look at an example where gridIdx.x = 6, gridIdx.y = 7, gridIdx.z = 1
(blockIdx.x, blockIdx.y)
mapping between threadblockID and C matrix:
-------------------------------------------------------
(0,0) | (0,1) | (0,2) | (0,3) | (0,4) | (0,5) | (0,6) |
-------------------------------------------------------
(1,0) | (1,1) | (1,2) | (1,3) | (1,4) | (1,5) | (1,6) |
-------------------------------------------------------
(2,0) | (2,1) | (2,2) | (2,3) | (2,4) | (2,5) | (2,6) |
-------------------------------------------------------
(3,0) | (3,1) | (3,2) | (3,3) | (3,4) | (3,5) | (3,6) |
-------------------------------------------------------
(4,0) | (4,1) | (4,2) | (4,3) | (4,4) | (4,5) | (4,6) |
-------------------------------------------------------
(5,0) | (5,1) | (5,2) | (5,3) | (5,4) | (5,5) | (5,6) |
-------------------------------------------------------

A ColumnMajorBlockSwizzle<1, OneDirection> will imply the above order where threadblocks are
launched in a column major

A ColumnMajorBlockSwizzle<2, OneDirection> swizzles things a little,
-------------------------------------------------------
(0,0) | (3,0) | (0,2) | (3,2) | (0,4) | (3,4) | (0,6) |
-------------------------------------------------------
(0,1) | (3,1) | (0,3) | (3,3) | (0,5) | (3,5) | (1,6) |
-------------------------------------------------------
(1,0) | (4,0) | (1,2) | (4,2) | (1,4) | (4,4) | (2,6) |
-------------------------------------------------------
(1,1) | (4,1) | (1,3) | (4,3) | (1,5) | (4,5) | (3,6) |
-------------------------------------------------------
(2,0) | (5,0) | (2,2) | (5,2) | (2,4) | (5,4) | (4,6) |
-------------------------------------------------------
(2,1) | (5,1) | (2,3) | (5,3) | (2,5) | (5,5) | (5,6) |
-------------------------------------------------------

so in memory, it would apprear that we work on 2 columns at a time rather than 1
Note that the index here really represent how each block maps to memory

A ColumnMajorBlockSwizzle<1, Boustrophedon> is similar to ColumnMajorBlockSwizzle<1, OneDirection>
except that every column flips the ordering against the previous one
-------------------------------------------------------
(0,0) | (5,1) | (0,2) | (5,3) | (0,4) | (5,5) | (0,6) |
-------------------------------------------------------
(1,0) | (4,1) | (1,2) | (4,3) | (1,4) | (4,5) | (1,6) |
-------------------------------------------------------
(2,0) | (3,1) | (2,2) | (3,3) | (2,4) | (3,5) | (2,6) |
-------------------------------------------------------
(3,0) | (2,1) | (3,2) | (2,3) | (3,4) | (2,5) | (3,6) |
-------------------------------------------------------
(4,0) | (1,1) | (4,2) | (1,3) | (4,4) | (1,5) | (4,6) |
-------------------------------------------------------
(5,0) | (0,1) | (5,2) | (0,3) | (5,4) | (0,5) | (5,6) |
-------------------------------------------------------

similarily, A ColumnMajorBlockSwizzle<2, Boustrophedon> looks like
-------------------------------------------------------
(0,0) | (3,0) | (2,3) | (5,3) | (0,4) | (3,4) | (5,6) |
-------------------------------------------------------
(0,1) | (3,1) | (2,2) | (5,2) | (0,5) | (3,5) | (4,6) |
-------------------------------------------------------
(1,0) | (4,0) | (1,3) | (4,3) | (1,4) | (4,4) | (3,6) |
-------------------------------------------------------
(1,1) | (4,1) | (1,2) | (4,2) | (1,5) | (4,5) | (2,6) |
-------------------------------------------------------
(2,0) | (5,0) | (0,3) | (3,3) | (2,4) | (5,4) | (1,6) |
-------------------------------------------------------
(2,1) | (5,1) | (0,2) | (3,2) | (2,5) | (5,5) | (0,6) |
-------------------------------------------------------

*/

template <int groupCols, enum swizzleDirection::Kind swDirection>
struct ColumnMajorBlockSwizzle {
  /// Ctor.
  CUTLASS_HOST_DEVICE ColumnMajorBlockSwizzle() {}

  /// Swizzle the block index.
  CUTLASS_DEVICE dim3 swizzle() {
    assert(gridDim.z == 1);
    int linearIdx = getLinearIdx<swDirection>(groupCols);
    dim3 swizzledBlockIdx;
    int currGroupCols = groupCols;
    int prevGroupCols = groupCols;

    if ((gridDim.y % groupCols != 0) && ((blockIdx.y + (gridDim.y % groupCols)) >= gridDim.y)) {
      // last colmuns if gridDim.y is not divisble by groupCols
      currGroupCols = gridDim.y % groupCols;
    }

    swizzledBlockIdx.x = (linearIdx / currGroupCols) % gridDim.x;
    swizzledBlockIdx.y =
        linearIdx % currGroupCols + prevGroupCols * (linearIdx / (prevGroupCols * gridDim.x));
    swizzledBlockIdx.z = blockIdx.z;

    return swizzledBlockIdx;
  }

  ///
  CUTLASS_HOST_DEVICE dim3 get_grid_layout(GemmCoord const &problem_size,
                                           Coord<3> const &OutputTile) {
    dim3 grid;
    grid.x = (problem_size.m() + OutputTile[2] - 1) / OutputTile[2];
    grid.y = (problem_size.n() + OutputTile[1] - 1) / OutputTile[1];
    grid.z = problem_size.batch();
    return grid;
  }

  ///
  CUTLASS_DEVICE Coord<3> get_threadblock_offset(Coord<3> const &OutputTile) {
    dim3 block = swizzle();
    Coord<3> threadblock_offset =
        make_Coord(0, block.y * OutputTile[1], block.x * OutputTile[2]);
    return threadblock_offset;
  }

  ///
  CUTLASS_DEVICE int get_batch_id() {
    dim3 block = swizzle();
    return block.z;
  }

  /// check if at the last partition
  CUTLASS_DEVICE bool is_last_partition() {
    if (get_batch_id() == (gridDim.z - 1))
      return true;
    else
      return false;
  }

  ///
  CUTLASS_DEVICE Coord<3> get_threadblock_bounds(GemmCoord const &problem_size,
                                                 int partitionK_range) {
    // every partition except the last one has a smaller range
    // partitionK_range is the bounds for every partition except the last one
    // the last partition's bounds is the same with problem size
    if (is_last_partition())
      return problem_size.knm();
    else
      return make_Coord(partitionK_range, problem_size.n(), problem_size.m());
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/*

consider a regular thread block mapping btween matrix C and different thread blocks
note that C is column major, and the leading dimension of thread block id is blockIdx.x

let's look at an example where gridIdx.x = 6, gridIdx.y = 7, gridIdx.z = 1
(blockIdx.x, blockIdx.y)
mapping between threadblockID and C matrix:
-------------------------------------------------------
(0,0) | (0,1) | (0,2) | (0,3) | (0,4) | (0,5) | (0,6) |
-------------------------------------------------------
(1,0) | (1,1) | (1,2) | (1,3) | (1,4) | (1,5) | (1,6) |
-------------------------------------------------------
(2,0) | (2,1) | (2,2) | (2,3) | (2,4) | (2,5) | (2,6) |
-------------------------------------------------------
(3,0) | (3,1) | (3,2) | (3,3) | (3,4) | (3,5) | (3,6) |
-------------------------------------------------------
(4,0) | (4,1) | (4,2) | (4,3) | (4,4) | (4,5) | (4,6) |
-------------------------------------------------------
(5,0) | (5,1) | (5,2) | (5,3) | (5,4) | (5,5) | (5,6) |
-------------------------------------------------------

A RowMajorBlockSwizzle<1, OneDirection> will effectively transpose the map

-----------------------------------------------
(0,0) | (1,0) | (2,0) | (3,0) | (4,0) | (5,0) |
-----------------------------------------------
(0,1) | (1,1) | (2,1) | (3,1) | (4,1) | (5,1) |
-----------------------------------------------
(0,2) | (1,2) | (2,2) | (3,2) | (4,2) | (5,2) |
-----------------------------------------------
(0,3) | (1,3) | (2,3) | (3,3) | (4,3) | (5,3) |
-----------------------------------------------
(0,4) | (1,4) | (2,4) | (3,4) | (4,4) | (5,4) |
---------------------------------------------
(0,5) | (1,5) | (2,5) | (3,5) | (4,5) | (5,5) |
-----------------------------------------------
(0,6) | (1,6) | (2,6) | (3,6) | (4,6) | (5,6) |
-----------------------------------------------

It would aprear in memory we are working on 1 row at a time

A ColumnMajorBlockSwizzle<2, OneDirection> swizzles things a little bit more
-----------------------------------------------
(0,0) | (1,3) | (2,0) | (3,3) | (4,0) | (5,3) |
-----------------------------------------------
(1,0) | (0,4) | (3,0) | (2,4) | (5,0) | (4,4) |
-----------------------------------------------
(0,1) | (1,4) | (2,1) | (3,4) | (4,1) | (5,4) |
-----------------------------------------------
(1,1) | (0,5) | (3,1) | (2,5) | (5,1) | (4,5) |
-----------------------------------------------
(0,2) | (1,5) | (2,2) | (3,5) | (4,2) | (5,5) |
---------------------------------------------
(1,2) | (0,6) | (3,2) | (2,6) | (5,2) | (4,6) |
-----------------------------------------------
(0,3) | (1,6) | (2,3) | (3,6) | (4,3) | (5,6) |
-----------------------------------------------

so in memory, it would apprear that we work on 2 rows at a time rather than 1 row
Note that the index here really represent how each block maps to memory

A RowMajorBlockSwizzle<1, Boustrophedon> is similar to RowMajorBlockSwizzle<1, OneDirection>
except that every column flips the ordering against the previous one

-----------------------------------------------
(0,0) | (1,6) | (2,0) | (3,6) | (4,0) | (5,6) |
-----------------------------------------------
(0,1) | (1,5) | (2,1) | (3,5) | (4,1) | (5,5) |
-----------------------------------------------
(0,2) | (1,4) | (2,2) | (3,4) | (4,2) | (5,4) |
-----------------------------------------------
(0,3) | (1,3) | (2,3) | (3,3) | (4,3) | (5,3) |
-----------------------------------------------
(0,4) | (1,2) | (2,4) | (3,2) | (4,4) | (5,2) |
---------------------------------------------
(0,5) | (1,1) | (2,5) | (3,1) | (4,5) | (5,1) |
-----------------------------------------------
(0,6) | (1,0) | (2,6) | (3,0) | (4,6) | (5,0) |
-----------------------------------------------

similarily, A RowMajorBlockSwizzle<2, Boustrophedon> looks like
-----------------------------------------------
(0,0) | (1,3) | (2,3) | (3,6) | (4,0) | (5,3) |
-----------------------------------------------
(1,0) | (0,4) | (3,2) | (2,6) | (5,0) | (4,4) |
-----------------------------------------------
(0,1) | (1,4) | (2,2) | (3,5) | (4,1) | (5,4) |
-----------------------------------------------
(1,1) | (0,5) | (3,1) | (2,5) | (5,1) | (4,5) |
-----------------------------------------------
(0,2) | (1,5) | (2,1) | (3,4) | (4,2) | (5,5) |
---------------------------------------------
(1,2) | (0,6) | (3,0) | (2,4) | (5,2) | (4,6) |
-----------------------------------------------
(0,3) | (1,6) | (2,0) | (3,3) | (4,3) | (5,6) |
-----------------------------------------------

*/

template <int groupRows, enum swizzleDirection::Kind swDirection>
struct RowMajorBlockSwizzle {
  /// Ctor.
  CUTLASS_HOST_DEVICE RowMajorBlockSwizzle() {}

  /// Swizzle the block index.
  CUTLASS_DEVICE dim3 swizzle() {
    assert(gridDim.z == 1);
    int linearIdx = getLinearIdx<swDirection>(groupRows);
    dim3 swizzledBlockIdx;
    int currGroupRows = groupRows;
    int prevGroupRows = groupRows;

    if ((gridDim.y % groupRows != 0) && ((blockIdx.y + (gridDim.y % groupRows)) >= gridDim.y)) {
      // last columns
      currGroupRows = gridDim.y % groupRows;
    }

    swizzledBlockIdx.x =
        linearIdx % currGroupRows + prevGroupRows * (linearIdx / (prevGroupRows * gridDim.x));
    swizzledBlockIdx.y = (linearIdx / currGroupRows) % gridDim.x;
    swizzledBlockIdx.z = blockIdx.z;

    return swizzledBlockIdx;
  }

  ///
  CUTLASS_HOST_DEVICE dim3 get_grid_layout(GemmCoord const &problem_size,
                                           Coord<3> const &OutputTile) {
    dim3 grid;
    grid.x = (problem_size.n() + OutputTile[1] - 1) / OutputTile[1];
    grid.y = (problem_size.m() + OutputTile[2] - 1) / OutputTile[2];
    grid.z = problem_size.batch();
    return grid;
  }

  ///
  CUTLASS_DEVICE Coord<3> get_threadblock_offset(Coord<3> const &OutputTile) {
    dim3 block = swizzle();
    Coord<3> threadblock_offset =
        make_Coord(0, block.y * OutputTile[1], block.x * OutputTile[2]);
    return threadblock_offset;
  }

  ///
  CUTLASS_DEVICE int get_batch_id() {
    dim3 block = swizzle();
    return block.z;
  }

  /// check if at the last partition
  CUTLASS_DEVICE bool is_last_partition() {
    if (get_batch_id() == (gridDim.z - 1) )
      return true;
    else
      return false;
  }

  ///
  CUTLASS_DEVICE Coord<3> get_threadblock_bounds(GemmCoord const &problem_size,
                                                 int partitionK_range) {
    // every partition except the last one has a smaller range
    // partitionK_range is the bounds for every partition except the last one
    // the last partition's bounds is the same with problem size
    if (is_last_partition())
      return problem_size.knm();
    else
      return make_Coord(partitionK_range, problem_size.n(), problem_size.m());
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
