/***************************************************************************************************
 * Copyright (c) 2017-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright notice, this list of
 *     conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice, this list of
 *     conditions and the following disclaimer in the documentation and/or other materials
 *     provided with the distribution.
 *   * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without specific prior written
 *     permission.
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
  \brief Defines properties of GEMM computation that impose some constraints on caller.
*/
#pragma once

#include "cutlass/shape.h"

namespace cutlass {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// The scalar type for A.
    typename ScalarA_,
    /// The scalar type for B.
    typename ScalarB_,
    /// The scalar type for C.
    typename ScalarC_,
    /// The scalar type for D.
    typename ScalarD_,
    /// The threadblock tile size for the GEMM KxNxM.
    typename OutputTile_,
    /// The functor to do the math.
    typename MultiplyAdd_,
    /// The number of scalars per LDG for A.
    int kScalarsPerLdgA_,
    /// The number of scalars per STS for A.
    int kScalarsPerStsA_,
    /// The number of scalars per LDG for A.
    int kScalarsPerLdsA_,
    /// The number of scalars per LDG for B.
    int kScalarsPerLdgB_,
    /// The number of scalars per STS for B.
    int kScalarsPerStsB_,
    /// The number of scalars per LDS for B.
    int kScalarsPerLdsB_,
    /// The number of scalars per LDG for C and STG for D.
    int kScalarsPerLdgCAndStgD_,
    /// The number of scalars per STS for D.
    int kScalarsPerStsD_,
    /// The number of scalars per LDS for D.
    int kScalarsPerLdsD_,
    /// The number of stages in shared memory to do single/double/triple-buffering.
    int kStages_,
    /// If true, residue is computed in mainloop. If false, separate loops are instantiated.
    bool kResidueSeparate_ = false,
    /// Is residue performed in prologue?
    bool kResidueInProlog_ = false,
    /// If true, kernel is launched with CUDA launch bounds specified
    bool kLaunchBounds_ = true>
struct GemmConfig {
  //
  /// The scalar for A.
  typedef ScalarA_ ScalarA;
  /// The scalar for B.
  typedef ScalarB_ ScalarB;
  /// The scalar for C.
  typedef ScalarC_ ScalarC;
  /// The scalar for D.
  typedef ScalarD_ ScalarD;

  /// The tile.
  typedef OutputTile_ OutputTile;
  /// The functor to do D = A*B + C.
  typedef MultiplyAdd_ MultiplyAdd;
  /// The shape of the instruction.
  typedef typename MultiplyAdd::InstructionShape InstructionShape;
  /// The shape of warp-level GEMM
  typedef typename MultiplyAdd::AccumulatorsPerWarp AccumulatorsPerWarp;
  /// The accumulators.
  typedef typename MultiplyAdd::Accumulators Accumulators;

  /// The number of warps.
  typedef typename ShapeDiv<OutputTile, AccumulatorsPerWarp>::Shape Warps;
  /// The default warp size (32 threads per warp).
  static int const kWarpSize = cutlass::kWarpSize;
  /// The numnber of threads.
  static int const kThreads = ShapeCount<Warps>::kCount * kWarpSize;

  /// The number of scalars per LDG/STS/LDS for A.
  static int const kScalarsPerLdgA = kScalarsPerLdgA_;
  static int const kScalarsPerStsA = kScalarsPerStsA_;
  static int const kScalarsPerLdsA = kScalarsPerLdsA_;

  /// The number of scalars per LDG/STS/LDS for B.
  static int const kScalarsPerLdgB = kScalarsPerLdgB_;
  static int const kScalarsPerStsB = kScalarsPerStsB_;
  static int const kScalarsPerLdsB = kScalarsPerLdsB_;

  /// The number of scalars per LDG for C.
  static int const kScalarsPerLdgC = kScalarsPerLdgCAndStgD_;

  /// The number of scalars per STS/LDS/STG for D.
  static int const kScalarsPerStgD = kScalarsPerLdgCAndStgD_;
  static int const kScalarsPerStsD = kScalarsPerStsD_;
  static int const kScalarsPerLdsD = kScalarsPerLdsD_;

  /// The number of accumulators that are going to be fed from one LDS A/B.
  static int const kAccumulatorsPerLdsA = kScalarsPerLdsA / InstructionShape::kD;
  static int const kAccumulatorsPerLdsB = kScalarsPerLdsB / InstructionShape::kD;

  /// The number of stages in shared memory to implement double, triple, more-buffering.
  static int const kStages = kStages_;

  /// If true, mainloop is instantiated twice. The first instantiation contains no predicate
  // updates and is more efficient for some kernels. If false, only a single mainloop is
  // instantaited.
  static bool const kResidueSeparate = kResidueSeparate_;

  /// If true, residue is computed in the prologue.
  static bool const kResidueInProlog = kResidueInProlog_;

  /// If true, kernel is launched with launch bounds specified
  static bool const kLaunchBounds = kLaunchBounds_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace gemm
}  // namespace cutlass
