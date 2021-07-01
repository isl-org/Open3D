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
    \brief Implements warp-level multiply-accumulate operations using Volta's mma.sync instruction
*/

#pragma once

#include "cutlass/arch/mma.h"
#include "cutlass/fragment.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Shape of a warp-level GEMM (K-by-N-by-M)
    typename WarpGemmShape_,
    /// Layout of A multiplicand
    MatrixLayout::Kind LayoutA,
    /// Data type of A multiplicand
    typename ScalarA_,
    /// Layout of B multiplicand
    MatrixLayout::Kind LayoutB,
    /// Data type of A multiplicand
    typename ScalarB_,
    /// Data type of accumulators
    typename ScalarC_>
struct Volta884MultiplyAdd {
  //
  // Constant and type definitions
  //

  /// Shape of a warp-level GEMM (K-by-N-by-M)
  typedef WarpGemmShape_ WarpGemmShape;

  /// Shape of a warp-level GEMM (K-by-N-by-M)
  typedef WarpGemmShape_ AccumulatorsPerWarp;

  /// Most of the Volta884 code assumes interleaved 32x32 tiles
  typedef Shape<4, 32, 32> InterleavedTileShape;

  /// Shape of an individual warp-wide Volta mma.sync instruction
  typedef Shape<4, 16, 16> InstructionShape;

  /// Shape of a warp-level matrix multiply operation
  typedef Shape<InstructionShape::kD, WarpGemmShape::kH, WarpGemmShape::kW> WarpTile;

  /// Verify WarpTile is a multiple of fundamental 32x32 interleaved tile
  static_assert(!(WarpTile::kH % InterleavedTileShape::kH) &&
                    !(WarpTile::kW % InterleavedTileShape::kW) && WarpTile::kD == 4,
                "WarpTile must be a multiple of InterleavedTileShape.");

  /// Layout of A multiplicand
  static MatrixLayout::Kind const kLayoutA = LayoutA;
  /// Layout of B multiplicand
  static MatrixLayout::Kind const kLayoutB = LayoutB;

  /// The type for A.
  typedef ScalarA_ ScalarA;
  /// The type for B.
  typedef ScalarB_ ScalarB;
  /// The type for C and D.
  typedef ScalarC_ ScalarC;

  /// Hard-coded comptue type supported on Volta
  static arch::ComputeType::Kind const kComputeType = arch::ComputeType::kDefault;

  /// Defines a warp-level matrix multiply-accumulate operation performed by a warp.
  //
  // The layout is as follows. The entire warp performs a 64x64x4 GEMM using Volta mma.sync macros
  // arranged as a 2x2 tile of adjacent, 32x32x4 matrix products. These are implemented as a
  // 2x2 arrangement of spatially interleaved Volta mma.sync macros.
  //
  // The Iterations shape maps to the following dimensions of the above warp-level GEMM:
  //
  //   kC: number of rows of Volta mma.sync macros in 32x32x4 tile
  //   kW: number of columns of Volta mma.sync macros in 32x32x4 tile
  //   kH: number of rows of 32x32x4 macros in larger 64x64x4 tile
  //   kD: number of columns of 32x32x4 macros in larger 64x64x4 tile
  //
  // A column-major ordering would arrange C and H as the inner-most loops, with W and D as the
  // outer-most.
  //
  typedef Shape<WarpTile::kH / InterleavedTileShape::kH,
                WarpTile::kW / InterleavedTileShape::kW,
                InterleavedTileShape::kH / InstructionShape::kH,
                InterleavedTileShape::kW / InstructionShape::kW>
      Iterations;

  /// Number of multiplicand elements per instruction
  static int const kMultElementsPerInst = 4;

  /// Number of multiplicand elements per instruction
  static int const kAccumElementsPerInst = 8;

  /// Fragment definition for A multiplicand
  typedef Fragment<ScalarA, Iterations::kH * Iterations::kC * kMultElementsPerInst> FragmentA;

  /// Fragment definition for B multiplicand
  typedef Fragment<ScalarB, Iterations::kW * Iterations::kD * kMultElementsPerInst> FragmentB;

  /// Fragment definition for accumulators
  typedef Fragment<ScalarC, ShapeCount<Iterations>::kCount * kAccumElementsPerInst> Accumulators;

  //
  // Methods
  //

  /// Ctor.
  CUTLASS_DEVICE Volta884MultiplyAdd() {}

  /// Multiply : d = (-)a*b + c.
  CUTLASS_DEVICE void multiply_add(FragmentA const& A,
                                   FragmentB const& B,
                                   Accumulators const& C,
                                   Accumulators& D,
                                   bool negate = false) {
// Guard conditional needed for __hneg2
#if (__CUDA_ARCH__ >= 700 && __CUDA_ARCH__ <= 750 && CUTLASS_ENABLE_TENSOR_CORE_MMA)

    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {  // Outer column
      CUTLASS_PRAGMA_UNROLL
      for (int w = 0; w < Iterations::kW; ++w) {  // Inner column
        CUTLASS_PRAGMA_UNROLL
        for (int h_raw = 0; h_raw < Iterations::kH; ++h_raw) {  // Outer row
          CUTLASS_PRAGMA_UNROLL
          for (int c_raw = 0; c_raw < Iterations::kC; ++c_raw) {  // Inner row

            int op_col = (w + Iterations::kW * d);

            // Column-major serpentine sequence to maximize reuse of B operand.
            int h = h_raw;
            int c = c_raw;

            if (op_col & 1) {
              h = Iterations::kH - h_raw - 1;
              c = Iterations::kC - c_raw - 1;
            }

            int op_row = (c + Iterations::kC * h);
            int op_idx = c + Iterations::kC * (w + Iterations::kW * (h + Iterations::kH * d));

            ScalarA operand_A[kMultElementsPerInst];

            reinterpret_cast<uint64_t&>(operand_A[0]) =
                reinterpret_cast<uint64_t const&>(A[op_row * kMultElementsPerInst]);

            if (negate) {
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < kMultElementsPerInst; i += 2) {
                reinterpret_cast<half2&>(operand_A[i]) =
                    __hneg2(reinterpret_cast<half2 const&>(A[op_row * kMultElementsPerInst + i]));
              }
            }

            // Issue a Volta mma.sync instruction
            arch::mma<InstructionShape,
                      kLayoutA,
                      ScalarA,
                      kLayoutB,
                      ScalarB,
                      ScalarC,
                      kComputeType>(

                operand_A, //&A[op_row * kMultElementsPerInst],
                &B[op_col * kMultElementsPerInst],
                &C[op_idx * kAccumElementsPerInst],
                &D[op_idx * kAccumElementsPerInst]);
          }
        }
      }
    }
#endif  // if (__CUDA_ARCH__ >= 700 && __CUDA_ARCH__ <=750 && CUTLASS_ENABLE_TENSOR_CORE_MMA)
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Accumulator, typename WarpDelta, typename Iterations>
struct Volta884NaiveEpilogue;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive epilogue specialized for f32 accumulators - may be considered authoritative mapping of
/// accumulators to mma.sync operations.
template <typename WarpDelta_, typename Iterations_>
struct Volta884NaiveEpilogue<float, WarpDelta_, Iterations_> {
  /// Accumulator data type
  typedef float ScalarC;

  /// Output accumulator type
  typedef float ScalarD;

  /// BLAS Scalar type
  typedef float Scalar;

  /// Delta among warp tiles
  typedef WarpDelta_ WarpDelta;

  /// Number of Volta mma.sync operations
  typedef Iterations_ Iterations;

  /// Most of the Volta884 code assumes interleaved 32x32 tiles
  typedef Shape<4, 32, 32> InterleavedTileShape;

  /// Number of multiplicand elements per instruction
  static int const kAccumElementsPerInst = 8;

  /// Fragment definition for accumulators
  typedef Fragment<ScalarC, ShapeCount<Iterations>::kCount * kAccumElementsPerInst> Accumulators;

  /// Params object
  struct Params {
    /// Pointer to output matrix
    ScalarC* ptr;

    /// stride
    int ldm;

    /// Scalar alpha
    float alpha;

    /// Scalar beta
    float beta;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() : ptr(0), ldm(0), alpha(1), beta(0) {}

    CUTLASS_HOST_DEVICE
    Params(ScalarC* _ptr, int _ldm, float _alpha = 1, float _beta = 0)
        : ptr(_ptr), ldm(_ldm), alpha(_alpha), beta(_beta) {}

    /// Initialize method
    CUTLASS_HOST_DEVICE
    int initialize(ScalarC* _ptr, int _ldm, float _alpha = 1, float _beta = 0) {
      ptr = _ptr;
      ldm = _ldm;
      alpha = _alpha;
      beta = _beta;
      return 0;
    }

    template <typename GemmDesc_>
    CUTLASS_HOST_DEVICE int initialize(GemmDesc_ const& desc) {
      ptr = reinterpret_cast<ScalarC*>(desc.D.data());
      ldm = desc.D.leading_dim();
      alpha = desc.alpha;
      beta = desc.beta;
      return 0;
    }
  };

  /// Shared stoarge
  struct SharedStorage {};

  /// Helper used to compute initial offset for each thread
  struct InitialOffset {
    int row_offset;
    int col_offset;

    /// Constructor
    CUTLASS_DEVICE
    InitialOffset() {
      int warp_id = (threadIdx.x >> 5);
      int lane_id = (threadIdx.x & 0x1f);
      int quad_id = (lane_id >> 2);
      int quadpair_id = (quad_id & 0x3);

      int quadpair_row = (quadpair_id & 1);
      int quadpair_col = (quadpair_id >> 1);
      int quad_hilo = (quad_id >> 2) & 1;

      // compute initial offset
      int warp_row_offset = (warp_id % WarpDelta::kW) * InterleavedTileShape::kW;
      int warp_col_offset = (warp_id / WarpDelta::kW) * InterleavedTileShape::kH;

      int thread_row_offset = (quadpair_row * 2 + quad_hilo) * 8 + (lane_id & 1);
      int thread_col_offset = (quadpair_col * 2) * 8 + (lane_id & 2);

      row_offset = warp_row_offset + thread_row_offset;
      col_offset = warp_col_offset + thread_col_offset;
    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  /// Problem size
  Coord<3> problem_size;

  //
  // Methods
  //

  /// Computes initial offset for each thread
  CUTLASS_DEVICE Volta884NaiveEpilogue(Params const& _params,
                                       Coord<3> const& _problem_size = make_Coord(1024, 1024, 1024))
      : params(_params), problem_size(_problem_size) {}

  /// Computes initial offset for each thread
  CUTLASS_DEVICE Volta884NaiveEpilogue(ScalarC* _ptr,
                                       int _ldm,
                                       Coord<3> const& _problem_size = make_Coord(1024, 1024, 1024))
      : params(_ptr, _ldm), problem_size(_problem_size) {}

  /// Computes initial offset for each thread
  CUTLASS_DEVICE Volta884NaiveEpilogue(Params const& _params,
                                       SharedStorage& shared_storage,
                                       Coord<3> const& _problem_size = make_Coord(1024, 1024, 1024))
      : params(_params), problem_size(_problem_size) {}

  /// Sets accumulators to zero
  CUTLASS_DEVICE void clear(Accumulators& C) {
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < Iterations::kC; ++c) {
            int op_idx = c + Iterations::kC * (w + Iterations::kW * (h + Iterations::kH * d));

            CUTLASS_PRAGMA_UNROLL
            for (int reg = 0; reg < kAccumElementsPerInst; ++reg) {
              C[op_idx * kAccumElementsPerInst + reg] = 0;
            }
          }
        }
      }
    }
  }

  /// Naive load operation for debugging
  CUTLASS_DEVICE void load(Accumulators& C,
                           Coord<3> const& threadblock_offset = make_Coord(0, 0, 0)) {
    InitialOffset initial;

    initial.row_offset += threadblock_offset[2];
    initial.col_offset += threadblock_offset[1];

    ScalarC const* load_ptr = params.ptr + initial.row_offset + params.ldm * initial.col_offset;

    // loads accumulators
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < Iterations::kC; ++c) {
            ScalarC const* op_ptr = load_ptr + h * WarpDelta::kW * InterleavedTileShape::kW +
                                    d * WarpDelta::kH * InterleavedTileShape::kH * params.ldm;

            int op_idx = c + Iterations::kC * (w + Iterations::kW * (h + Iterations::kH * d));

            CUTLASS_PRAGMA_UNROLL
            for (int reg = 0; reg < kAccumElementsPerInst; ++reg) {
              int tr = (reg & 2) + c * 4;
              int tc = (reg & 1) + (reg & 4) * 2 + w * 4;

              int row = initial.row_offset + h * WarpDelta::kW * InterleavedTileShape::kW + tr;
              int column = initial.col_offset + d * WarpDelta::kH * InterleavedTileShape::kH + tc;

              if (row < problem_size[2] && column < problem_size[1]) {
                C[op_idx * kAccumElementsPerInst + reg] = op_ptr[tr + tc * params.ldm];
              }
            }
          }
        }
      }
    }
  }

  /// Naive store operation for debugging
  CUTLASS_DEVICE void store(Accumulators const& C,
                            Coord<3> const& threadblock_offset = make_Coord(0, 0, 0)) {
    InitialOffset initial;

    initial.row_offset += threadblock_offset[2];
    initial.col_offset += threadblock_offset[1];

    ScalarC* store_ptr = params.ptr + initial.row_offset + params.ldm * initial.col_offset;

    // store out accumulators
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < Iterations::kC; ++c) {
            ScalarC* op_ptr = store_ptr + h * WarpDelta::kW * InterleavedTileShape::kW +
                              d * WarpDelta::kH * InterleavedTileShape::kH * params.ldm;

            int op_idx = c + Iterations::kC * (w + Iterations::kW * (h + Iterations::kH * d));

            CUTLASS_PRAGMA_UNROLL
            for (int reg = 0; reg < kAccumElementsPerInst; ++reg) {
              int tr = (reg & 2) + c * 4;
              int tc = (reg & 1) + (reg & 4) * 2 + w * 4;

              int row = initial.row_offset + h * WarpDelta::kW * InterleavedTileShape::kW + tr;
              int column = initial.col_offset + d * WarpDelta::kH * InterleavedTileShape::kH + tc;

              if (row < problem_size[2] && column < problem_size[1]) {
                op_ptr[tr + tc * params.ldm] =
                    params.alpha * C[op_idx * kAccumElementsPerInst + reg] +
                    params.beta * op_ptr[tr + tc * params.ldm];
              }
            }
          }
        }
      }
    }
  }

  /// CUTLASS Epilogue interface
  CUTLASS_DEVICE void epilogue(Accumulators const& C,
                               Coord<3> const& threadblock_offset = make_Coord(0, 0, 0)) {
    store(C, threadblock_offset);
  }

  CUTLASS_DEVICE void epilogue(Accumulators& C,
                               Coord<3> const& threadblock_offset = make_Coord(0, 0, 0)) {
    store(C, threadblock_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive epilogue specialized for f16 accumulators - may be considered authoritative mapping of
/// accumulators to mma.sync operations.
template <typename WarpDelta_, typename Iterations_>
struct Volta884NaiveEpilogue<half, WarpDelta_, Iterations_> {
  /// Accumulator data type
  typedef half ScalarC;

  /// Output accumulator type
  typedef half ScalarD;

  /// BLAS Scalar type
  typedef half Scalar;

  /// Delta among warp tiles
  typedef WarpDelta_ WarpDelta;

  /// Number of Volta mma.sync operations
  typedef Iterations_ Iterations;

  /// Most of the Volta884 code assumes interleaved 32x32 tiles
  typedef Shape<4, 32, 32> InterleavedTileShape;

  /// Number of multiplicand elements per instruction
  static int const kAccumElementsPerInst = 8;

  /// Fragment definition for accumulators
  typedef Fragment<ScalarC, ShapeCount<Iterations>::kCount * kAccumElementsPerInst> Accumulators;

  /// Params object
  struct Params {
    /// Pointer to output matrix
    ScalarC* ptr;

    /// stride
    int ldm;

    /// Scalar alpha
    half alpha;

    /// Scalar beta
    half beta;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() : ptr(0), ldm(0), alpha(1), beta(0) {}

    CUTLASS_HOST_DEVICE
    Params(ScalarC* _ptr, int _ldm, float _alpha = 1, float _beta = 0)
        : ptr(_ptr), ldm(_ldm), alpha(_alpha), beta(_beta) {}

    /// Initialize method
    CUTLASS_HOST_DEVICE
    int initialize(ScalarC* _ptr, int _ldm, float _alpha = 1, float _beta = 0) {
      ptr = _ptr;
      ldm = _ldm;
      alpha = _alpha;
      beta = _beta;
      return 0;
    }

    template <typename GemmDesc_>
    CUTLASS_HOST_DEVICE int initialize(GemmDesc_ const& desc) {
      ptr = reinterpret_cast<ScalarC*>(desc.D.data());
      ldm = desc.D.leading_dim();
      alpha = desc.alpha;
      beta = desc.beta;
      return 0;
    }
  };

  /// Shared stoarge
  struct SharedStorage {};

  /// Helper used to compute initial offset for each thread
  struct InitialOffset {
    int row_offset;
    int col_offset;

    /// Constructor
    CUTLASS_DEVICE
    InitialOffset() {
      int warp_id = (threadIdx.x >> 5);
      int lane_id = (threadIdx.x & 0x1f);
      int quad_id = (lane_id >> 2);
      int quadpair_id = (quad_id & 0x3);

      int quadpair_row = (quadpair_id & 1);
      int quadpair_col = (quadpair_id >> 1);
      int quad_hilo = (quad_id >> 2) & 1;

      // compute initial offset
      int warp_row_offset = (warp_id % WarpDelta::kW) * InterleavedTileShape::kW;
      int warp_col_offset = (warp_id / WarpDelta::kW) * InterleavedTileShape::kH;

      int thread_row_offset = (quadpair_row * 2 + quad_hilo) * 8 + (lane_id & 3);
      int thread_col_offset = (quadpair_col * 2) * 8;

      row_offset = warp_row_offset + thread_row_offset;
      col_offset = warp_col_offset + thread_col_offset;
    }
  };

  //
  // Data members
  //

  /// Parameters object
  Params params;

  /// Problem size
  Coord<3> problem_size;

  //
  // Methods
  //

  /// Computes initial offset for each thread
  CUTLASS_DEVICE Volta884NaiveEpilogue(Params const& _params)
      : params(_params), problem_size(make_Coord(1024, 1024, 1024)) {}

  /// Computes initial offset for each thread
  CUTLASS_DEVICE Volta884NaiveEpilogue(ScalarC* _ptr, int _ldm)
      : params(_ptr, _ldm), problem_size(make_Coord(1024, 1024, 1024)) {}

  /// Computes initial offset for each thread
  CUTLASS_DEVICE Volta884NaiveEpilogue(Params const& _params,
                                       SharedStorage& shared_storage,
                                       Coord<3> const& _problem_size = make_Coord(1024, 1024, 1024))
      : params(_params), problem_size(_problem_size) {}

  /// Sets accumulators to zero
  CUTLASS_DEVICE void clear(Accumulators& C) { C.clear(); }

  /// Naive load operation for debugging
  CUTLASS_DEVICE void load(Accumulators& C,
                           Coord<3> const& threadblock_offset = make_Coord(0, 0, 0)) {
    InitialOffset initial;

    initial.row_offset += threadblock_offset[2];
    initial.col_offset += threadblock_offset[1];

    ScalarC const* load_ptr = params.ptr + initial.row_offset + params.ldm * initial.col_offset;

    // loads accumulators
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < Iterations::kC; ++c) {
            ScalarC const* op_ptr = load_ptr + h * WarpDelta::kW * InterleavedTileShape::kW +
                                    d * WarpDelta::kH * InterleavedTileShape::kH * params.ldm;

            int op_idx = c + Iterations::kC * (w + Iterations::kW * (h + Iterations::kH * d));

            CUTLASS_PRAGMA_UNROLL
            for (int reg = 0; reg < kAccumElementsPerInst; ++reg) {
              int tr = c * 4;
              int tc = (reg & 3) + (reg & 4) * 2 + w * 4;

              int row = initial.row_offset + h * WarpDelta::kW * InterleavedTileShape::kW + tr;
              int column = initial.col_offset + d * WarpDelta::kH * InterleavedTileShape::kH + tc;

              if (row < problem_size[2] && column < problem_size[1]) {
                C[op_idx * kAccumElementsPerInst + reg] = op_ptr[tr + tc * params.ldm];
              }
            }
          }
        }
      }
    }
  }

  /// Naive store operation for debugging
  CUTLASS_DEVICE void store(Accumulators const& C,
                            Coord<3> const& threadblock_offset = make_Coord(0, 0, 0)) {
    InitialOffset initial;

    initial.row_offset += threadblock_offset[2];
    initial.col_offset += threadblock_offset[1];

    ScalarC* store_ptr = params.ptr + initial.row_offset + params.ldm * initial.col_offset;

    // store out accumulators
    CUTLASS_PRAGMA_UNROLL
    for (int d = 0; d < Iterations::kD; ++d) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < Iterations::kH; ++h) {
        CUTLASS_PRAGMA_UNROLL
        for (int w = 0; w < Iterations::kW; ++w) {
          CUTLASS_PRAGMA_UNROLL
          for (int c = 0; c < Iterations::kC; ++c) {
            ScalarC* op_ptr = store_ptr + h * WarpDelta::kW * InterleavedTileShape::kW +
                              d * WarpDelta::kH * InterleavedTileShape::kH * params.ldm;

            int op_idx = c + Iterations::kC * (w + Iterations::kW * (h + Iterations::kH * d));

            CUTLASS_PRAGMA_UNROLL
            for (int reg = 0; reg < kAccumElementsPerInst; ++reg) {
              int tr = c * 4;
              int tc = (reg & 3) + (reg & 4) * 2 + w * 4;

              int row = initial.row_offset + h * WarpDelta::kW * InterleavedTileShape::kW + tr;
              int column = initial.col_offset + d * WarpDelta::kH * InterleavedTileShape::kH + tc;

              if (row < problem_size[2] && column < problem_size[1]) {
                op_ptr[tr + tc * params.ldm] =
                    params.alpha * C[op_idx * kAccumElementsPerInst + reg] +
                    params.beta * op_ptr[tr + tc * params.ldm];
              }
            }
          }
        }
      }
    }
  }

  /// CUTLASS Epilogue interface
  CUTLASS_DEVICE void epilogue(Accumulators const& C,
                               Coord<3> const& threadblock_offset = make_Coord(0, 0, 0)) {
    store(C, threadblock_offset);
  }

  CUTLASS_DEVICE void epilogue(Accumulators& C,
                               Coord<3> const& threadblock_offset = make_Coord(0, 0, 0)) {
    store(C, threadblock_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
