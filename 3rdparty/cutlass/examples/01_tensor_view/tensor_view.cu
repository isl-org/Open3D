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
  This example demonstrates operations using TensorRef<> and TensorView<> as well as their explicit
  equivalent functionality in CUDA code.

  CUTLASS provides abstractions for interacting with multidimension tensors in device memory.
  Consequently, we define a hierarchy of pointer-like types for referencing tensors.

    T *                             - raw pointer to elements of type T

    cutlass::TensorRef<T, Rank>     - reference to a tensor of elements of type T and given rank.
                                      Includes a mapping function and associated stride vector for
                                      accessing elements in linear memory.

    cutlass::TensorView<T, Rank>:   - extends TensorRef<> by adding bounds information. This is a
      public TensorRef<T, Rank>       complete mathematical object which may be used as the argument
                                      to CUTLASS functions.

  The above provide an identity maping of a logical index space to linear memory. An element
  at logical coordinate X has an offset computed as follows:

     offset = dot(X, stride)

  where dot() computes the inner product of X and a vector of "strides."

  CUTLASS 1.1 introduces a mapping function and an additional 'rank' to offer a flexible way to
  map the logical index space of the tensor to memory. The mapping function maps a coordinate
  of rank R to an index space of rank S. The linear offset is computed as:

    offset = dot( MapFunc(X), stride )

  where stride is a vector of rank S.


  The complete template declaration for cutlass::TensorRef<> is as follows.

    template <
      /// Data type of element stored within tensor
      typename Storage,

      /// Rank of logical tensor
      int Rank,

      /// Maps a Coord<Rank> in the logical tensor index space to the internal n-D array
      typename MapFunc = IdentityTensorMapFunc<Rank>,

      /// Rank of internal n-D array
      int StorageRank_ = MapFunc::kStorageRank,

      /// Index type used for coordinates
      typename Index = int,

      /// Index type used for offsets and pointer differences
      typename LongIndex = long long
    >
    class TensorRef;


  CUTLASS kernels make extensive use of vectorization of memory accesses for efficiency and
  correctness. Consequently, we enforce a constraint on the strides used by mapping functions
  such that:

    1. The "fastest-changing" stride is always 1 thereby mandating that consecutive elements in
       that rank are consecutive in linear memory.

    2. The fastest changing rank is always last in the stride vector and not explicitly stored.

  Thus, the stride vector used by mapping functions has length of one fewer than the rank of the
  storage tensor. These constraints are consistent with the BLAS interface of passing matrices as
  a tuple consisting of a pointer and a "leading dimension." In fact, these are rank=2 tensors
  whose fastest changing dimension is 1, and the stride vector is of length 1.


  A typical mapping function might simply map the rows and columns of a matrix, a rank=2 tensor,
  to linear memory such that (1.) elements in the same column are consecutive in memory
  (column-major), or (2.) elements in the same row are consecutive (row-major). These can be
  accomplished by two different mapping functions whose stride vector is length=2. The first
  element is the "leading dimension."

  The following mapping functions demonstrates mappings for these canonical matrix layouts. In
  both cases, the logical index space is referenced by coordinates of the form (row, column).

  // cutlass/matrix_traits.h
  struct MatrixLayout {

    //
    // TensorRefMapFunc definitions for common layouts
    //

    /// Mapping function for row-major matrices
    struct RowMajor {

      /// Storage rank = 2 implies stride vector: (ldm, 1)
      static int const kStorageRank = 2;

      /// Maps (row, col) to (row, col)
      CUTLASS_HOST_DEVICE
      Coord<kStorageRank> operator()(Coord<2> const &coord) const {
        return coord;
      }
    };

    /// Mapping function for column-major matrices
    struct ColumnMajor {

      /// Storage rank = 2 implies stride vector: (ldm, 1)
      static int const kStorageRank = 2;

      /// Maps (row, col) to (col, row)
      CUTLASS_HOST_DEVICE
      Coord<kStorageRank> operator()(Coord<2> const &coord) const {
        return make_Coord(coord[1], coord[0]);
      }
    };
  };


  The requirement that the fastest-changing stride always be of unit size need not be a limitation.
  To implement "sparse" computations or matrix operations in which matrix elements have arbitrary
  stride along both row and column, define a mapping function whose storage rank is 3. This permits
  two elements of the stride vector to have a non-unit value. The map function defined in
  `cutlass::MatrixTraits::ContiguousLayout` is an example.

  ```
  /// Mapping function for scenario in which layout is row-major or column-major but this information
  /// is only available at runtime.
  struct ContiguousLayout {
    /// Arbitrary storage rank
    static int const kStorageRank = 3;

    /// Dimension of rows
    static int const kRow = 0;

    /// Dimension of columns
    static int const kColumn = 1;

    /// Mapping function defined by runtime variable. Returns coordinates in n-D storage array
    /// as (matrix row, matrix colum, 0)
    CUTLASS_HOST_DEVICE
    Coord<kStorageRank> operator()(MatrixCoord const &coord) const {
        return make_Coord(coord.row(), coord.column(), 0);
    }

    /// Helper to construct a stride vector based on contiguous matrix layout and leading dimension
    CUTLASS_HOST_DEVICE
    static Coord<kStorageRank> stride(MatrixLayout::Kind layout, int ldm) {
      if (layout == MatrixLayout::kRowMajor) {
        return make_Coord(ldm, 1, 1);
      }
      return make_Coord(1, ldm, 1);
    }
  };
  ```

  cutlass::TensorView<> extends this concept by including a size vector to specify the bounds of
  the index space. The value of each coordinate in the size vector defines the half-open range of
  indices whose smallest value is zero.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

// Standard Library includes
#include <iostream>
#include <vector>

//
// CUTLASS includes
//

// Defines cutlass::Coord<>
#include "cutlass/coord.h"

// Defines cutlass::TensorRef<>
#include "cutlass/tensor_ref.h"

// Defines cutlass::TensorView<>
#include "cutlass/tensor_view.h"

// Defines cutlass::MatrixLayout
#include "cutlass/matrix_traits.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Column-major matrix access
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a rank=2 tensor modeling a column-major matrix
typedef cutlass::TensorView<
  int,                                    // storage element is of type int
  2,                                      // tensor has rank=2 logical index space
  cutlass::MatrixLayout::ColumnMajor      // column-major mapping function
> TensorViewColumnMajor;

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to copy a matrix from raw memory into a cutlass::TensorView
__global__ void MatrixCopyColumnMajor(
  TensorViewColumnMajor destination,      // destination tensor accessed by TensorView
  int const *source,                      // source matrix accessed using cuBLAS-style pointer
  int ldm) {                              //   and leading dimension

  // Compute unique row and column for each thread
  int row = threadIdx.x + blockIdx.x * blockDim.x;
  int column = threadIdx.y + blockIdx.y * blockDim.y;

  // Define a coordinate based on the thread's row and column
  cutlass::Coord<2> coord = cutlass::make_Coord(row, column);

  // Bounds test
  if (coord < destination.size()) {

    // Access the element
    destination.at(coord) = source[row + column * ldm];
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Launches kernel MatrixCopyColumnMajor()
cudaError_t TestMatrixCopyColumnMajor() {
  cudaError_t result;

  int const M = 32;     // number of rows
  int const N = 16;     // number of columns

  int const ldm = 40;   // matrix leading dimension

  //
  // Allocate source and destination matrices
  //

  int *Destination;
  int *Source;

  int const matrix_capacity = ldm * N;                          // number of elements in memory needed to store matrix
  size_t const sizeof_matrix = sizeof(int) * matrix_capacity;   // size of matrix in bytes

  // Allocate destination and source matrices
  result = cudaMalloc((void **)&Destination, sizeof_matrix);
  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate destination matrix on device: " << cudaGetErrorString(result) << std::endl;
    return result;
  }

  result = cudaMalloc((void **)&Source, sizeof_matrix);
  if (result != cudaSuccess) {
    cudaFree(Destination);
    std::cerr << "Failed to allocate source matrix on device:" << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear destination matrix in device memory
  result = cudaMemset(Destination, 0, sizeof_matrix);
  if (result != cudaSuccess) {
    cudaFree(Destination);
    cudaFree(Source);
    std::cerr << "Failed to clear destination matrix: " << cudaGetErrorString(result) << std::endl;
    return result;
  }

  //
  // Initialize matrix
  //

  std::vector<int> source_host(matrix_capacity, 0);

  // Procedurally generate input results using several arbitrary constants.
  int const magic_row_stride = 2;
  int const magic_column_stride = 3;

  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      source_host.at(i + j * ldm) = i * magic_row_stride + j * magic_column_stride;
    }
  }

  // Copy to device memory
  result = cudaMemcpy(Source, source_host.data(), sizeof_matrix, cudaMemcpyHostToDevice);
  if (result != cudaSuccess) {
    cudaFree(Destination);
    cudaFree(Source);
    std::cerr << "Failed to copy from host to source matrix: " << cudaGetErrorString(result) << std::endl;
    return result;
  }

  //
  // Define a TensorView<> pointing to the destination matrix
  //
  TensorViewColumnMajor destination_view_device(
    Destination,                            // pointer to base of matrix in device memory
    cutlass::make_Coord(ldm, 1),            // stride vector
    cutlass::make_Coord(M, N)               // bounds of matrix
  );

  //
  // Launch kernel to copy matrix
  //

  dim3 block(16, 16);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  MatrixCopyColumnMajor<<< grid, block >>>(destination_view_device, Source, ldm);

  result = cudaGetLastError();
  if (result != cudaSuccess) {
    std::cerr << "Kernel MatrixCopyColumnMajor() failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(Destination);
    cudaFree(Source);

    return result;
  }

  //
  // Copy results to host memory
  //

  std::vector<int> dest_host(matrix_capacity, 0);

  result = cudaMemcpy(dest_host.data(), Destination, sizeof_matrix, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy destination matrix to host memory: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(Destination);
    cudaFree(Source);

    return result;
  }

  //
  // Verify result
  //

  // Define a TensorView for use in accessing host memory
  TensorViewColumnMajor destination_view_host(
    dest_host.data(),                          // pointer to base of matrix in host memory
    cutlass::make_Coord(ldm, 1),               // stride vector
    cutlass::make_Coord(M, N)                  // bounds of matrix
  );

  // Verify against procedurally computed results
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < M; ++i) {

      // computed result
      int expected = i * magic_row_stride + j * magic_column_stride;

      // access data by computing explicit offsets
      int got_explicit = dest_host.at(i + j * ldm);

      // access data in host memory through a TensorView
      int got_view = destination_view_host.at(cutlass::make_Coord(i, j));

      if (got_explicit != expected) {

        std::cerr << "Error at element (" << i << ", " << j
          << ") accessed through explicitly computed offset - expected: " << expected
          << ", got: " << got_explicit << std::endl;

        return cudaErrorUnknown;
      }

      if (got_view != expected) {

        std::cerr << "Error at element (" << i << ", " << j
          << ") accesed through TensorView<> on the host - expected: " << expected
          << ", got: " << got_view << std::endl;

        return cudaErrorUnknown;
      }
    }
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point for tensor_view example.
//
// usage:
//
//   02_tensor_view
//
int main() {

  cudaError_t result = TestMatrixCopyColumnMajor();

  if (result == cudaSuccess) {
    std::cout << "Passed" << std::endl;
  }

  return (result == cudaSuccess ? 0 : -1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
