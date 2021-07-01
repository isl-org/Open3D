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
    \brief Templates wrapping direct issue of MMA instructions to Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/shape.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Specifies internal data type for computation
struct ComputeType {
  enum Kind {
    kBegin,
    kDefault,   /// Compute type implied by operand and accumulator types
    kEnd
  };
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Direct wrapper for native MMA instruction
template <
    /// Warp-level matrix multiply-accumulate operation
    typename WmmaTile,
    /// Layout of A multiplicand
    MatrixLayout::Kind LayoutA,
    /// Data type of A multiplicand
    typename ScalarA,
    /// Layout of B multiplicand
    MatrixLayout::Kind LayoutB,
    /// Data type of A multiplicand
    typename ScalarB,
    /// Data type of accumulators
    typename ScalarC,
    /// Specifies particular compute type, overriding data types of operands
    ComputeType::Kind ComputeTy>
inline __device__ void mma(ScalarA const A[], ScalarB const B[], ScalarC const C[], ScalarC D[]);

/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

//
// 16x16x4
//

//
// FP16 accumulation
//

/// Volta mma.sync instruction
template <>
inline __device__ void mma<Shape<4, 16, 16>,
                           MatrixLayout::kRowMajor,
                           half,
                           MatrixLayout::kColumnMajor,
                           half,
                           half,
                           ComputeType::kDefault>(half const a[],
                                                  half const b[],
                                                  half const c[],
                                                  half d[]) {
#if (__CUDA_ARCH__ >= 700 && __CUDA_ARCH__ <= 750 && CUTLASS_ENABLE_TENSOR_CORE_MMA)

  unsigned const *A = reinterpret_cast<unsigned const *>(a);
  unsigned const *B = reinterpret_cast<unsigned const *>(b);
  unsigned const *C = reinterpret_cast<unsigned const *>(c);
  unsigned *D = reinterpret_cast<unsigned *>(d);

  asm volatile("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
  CUTLASS_ASSERT(0); // Collective matrix multiply instruction requires CUTLASS_ENABLE_TENSOR_CORE_MMA=1
#endif
}

/// Volta mma.sync instruction
template <>
inline __device__ void mma<Shape<4, 16, 16>,
                           MatrixLayout::kColumnMajor,
                           half,
                           MatrixLayout::kColumnMajor,
                           half,
                           half,
                           ComputeType::kDefault>(half const a[],
                                                  half const b[],
                                                  half const c[],
                                                  half d[]) {
#if (__CUDA_ARCH__ >= 700 && __CUDA_ARCH__ <= 750 && CUTLASS_ENABLE_TENSOR_CORE_MMA)

  unsigned const *A = reinterpret_cast<unsigned const *>(a);
  unsigned const *B = reinterpret_cast<unsigned const *>(b);
  unsigned const *C = reinterpret_cast<unsigned const *>(c);
  unsigned *D = reinterpret_cast<unsigned *>(d);

  asm volatile("mma.sync.aligned.m8n8k4.col.col.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
  CUTLASS_ASSERT(0); // Collective matrix multiply instruction requires CUTLASS_ENABLE_TENSOR_CORE_MMA=1
#endif
}

/// Volta mma.sync instruction
template <>
inline __device__ void mma<Shape<4, 16, 16>,
                           MatrixLayout::kRowMajor,
                           half,
                           MatrixLayout::kRowMajor,
                           half,
                           half,
                           ComputeType::kDefault>(half const a[],
                                                  half const b[],
                                                  half const c[],
                                                  half d[]) {
#if (__CUDA_ARCH__ >= 700 && __CUDA_ARCH__ <= 750 && CUTLASS_ENABLE_TENSOR_CORE_MMA)

  unsigned const *A = reinterpret_cast<unsigned const *>(a);
  unsigned const *B = reinterpret_cast<unsigned const *>(b);
  unsigned const *C = reinterpret_cast<unsigned const *>(c);
  unsigned *D = reinterpret_cast<unsigned *>(d);

  asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else
  CUTLASS_ASSERT(0); // Collective matrix multiply instruction requires CUTLASS_ENABLE_TENSOR_CORE_MMA=1
#endif
}

/// Volta mma.sync instruction
template <>
inline __device__ void mma<Shape<4, 16, 16>,
                           MatrixLayout::kColumnMajor,
                           half,
                           MatrixLayout::kRowMajor,
                           half,
                           half,
                           ComputeType::kDefault>(half const a[],
                                                  half const b[],
                                                  half const c[],
                                                  half d[]) {
#if (__CUDA_ARCH__ >= 700 && __CUDA_ARCH__ <= 750 && CUTLASS_ENABLE_TENSOR_CORE_MMA)

  unsigned const *A = reinterpret_cast<unsigned const *>(a);
  unsigned const *B = reinterpret_cast<unsigned const *>(b);
  unsigned const *C = reinterpret_cast<unsigned const *>(c);
  unsigned *D = reinterpret_cast<unsigned *>(d);

  asm volatile("mma.sync.aligned.m8n8k4.col.row.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};"
      : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));

#else 
  CUTLASS_ASSERT(0); // Collective matrix multiply instruction requires CUTLASS_ENABLE_TENSOR_CORE_MMA=1
#endif
}

//
// FP32 accumulation
//

/// Volta mma.sync instruction
template <>
inline __device__ void mma<Shape<4, 16, 16>,
                           MatrixLayout::kRowMajor,
                           half,
                           MatrixLayout::kColumnMajor,
                           half,
                           float,
                           ComputeType::kDefault>(half const a[],
                                                  half const b[],
                                                  float const C[],
                                                  float D[]) {
#if (__CUDA_ARCH__ >= 700 && __CUDA_ARCH__ <= 750 && CUTLASS_ENABLE_TENSOR_CORE_MMA)

  unsigned const *A = reinterpret_cast<unsigned const *>(a);
  unsigned const *B = reinterpret_cast<unsigned const *>(b);

  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%12,%13,%14,%15,%16,%17,%18,%19};"
      : "=f"(D[0]),
        "=f"(D[1]),
        "=f"(D[2]),
        "=f"(D[3]),
        "=f"(D[4]),
        "=f"(D[5]),
        "=f"(D[6]),
        "=f"(D[7])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]),
        "f"(C[4]),
        "f"(C[5]),
        "f"(C[6]),
        "f"(C[7]));

#else
  CUTLASS_ASSERT(0); // Collective matrix multiply instruction requires CUTLASS_ENABLE_TENSOR_CORE_MMA=1
#endif
}

/// Volta mma.sync instruction
template <>
inline __device__ void mma<Shape<4, 16, 16>,
                           MatrixLayout::kColumnMajor,
                           half,
                           MatrixLayout::kColumnMajor,
                           half,
                           float,
                           ComputeType::kDefault>(half const a[],
                                                  half const b[],
                                                  float const C[],
                                                  float D[]) {

#if (__CUDA_ARCH__ >= 700 && __CUDA_ARCH__ <= 750 && CUTLASS_ENABLE_TENSOR_CORE_MMA)

  unsigned const *A = reinterpret_cast<unsigned const *>(a);
  unsigned const *B = reinterpret_cast<unsigned const *>(b);

  asm volatile("mma.sync.aligned.m8n8k4.col.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%12,%13,%14,%15,%16,%17,%18,%19};"
      : "=f"(D[0]),
        "=f"(D[1]),
        "=f"(D[2]),
        "=f"(D[3]),
        "=f"(D[4]),
        "=f"(D[5]),
        "=f"(D[6]),
        "=f"(D[7])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]),
        "f"(C[4]),
        "f"(C[5]),
        "f"(C[6]),
        "f"(C[7]));

#else
  CUTLASS_ASSERT(0); // Collective matrix multiply instruction requires CUTLASS_ENABLE_TENSOR_CORE_MMA=1
#endif
}

/// Volta mma.sync instruction
template <>
inline __device__ void mma<Shape<4, 16, 16>,
                           MatrixLayout::kRowMajor,
                           half,
                           MatrixLayout::kRowMajor,
                           half,
                           float,
                           ComputeType::kDefault>(half const a[],
                                                  half const b[],
                                                  float const C[],
                                                  float D[]) {
#if (__CUDA_ARCH__ >= 700 && __CUDA_ARCH__ <= 750 && CUTLASS_ENABLE_TENSOR_CORE_MMA)

  unsigned const *A = reinterpret_cast<unsigned const *>(a);
  unsigned const *B = reinterpret_cast<unsigned const *>(b);

  asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%12,%13,%14,%15,%16,%17,%18,%19};"
      : "=f"(D[0]),
        "=f"(D[1]),
        "=f"(D[2]),
        "=f"(D[3]),
        "=f"(D[4]),
        "=f"(D[5]),
        "=f"(D[6]),
        "=f"(D[7])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]),
        "f"(C[4]),
        "f"(C[5]),
        "f"(C[6]),
        "f"(C[7]));

#else
  CUTLASS_ASSERT(0); // Collective matrix multiply instruction requires CUTLASS_ENABLE_TENSOR_CORE_MMA=1
#endif
}

/// Volta mma.sync instruction
template <>
inline __device__ void mma<Shape<4, 16, 16>,
                           MatrixLayout::kColumnMajor,
                           half,
                           MatrixLayout::kRowMajor,
                           half,
                           float,
                           ComputeType::kDefault>(half const a[],
                                                  half const b[],
                                                  float const C[],
                                                  float D[]) {
#if (__CUDA_ARCH__ >= 700 && __CUDA_ARCH__ <= 750 && CUTLASS_ENABLE_TENSOR_CORE_MMA)

  unsigned const *A = reinterpret_cast<unsigned const *>(a);
  unsigned const *B = reinterpret_cast<unsigned const *>(b);

  asm volatile ("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%12,%13,%14,%15,%16,%17,%18,%19};"
      : "=f"(D[0]),
        "=f"(D[1]),
        "=f"(D[2]),
        "=f"(D[3]),
        "=f"(D[4]),
        "=f"(D[5]),
        "=f"(D[6]),
        "=f"(D[7])
      : "r"(A[0]),
        "r"(A[1]),
        "r"(B[0]),
        "r"(B[1]),
        "f"(C[0]),
        "f"(C[1]),
        "f"(C[2]),
        "f"(C[3]),
        "f"(C[4]),
        "f"(C[5]),
        "f"(C[6]),
        "f"(C[7]));

#else
  CUTLASS_ASSERT(0); // Collective matrix multiply instruction requires CUTLASS_ENABLE_TENSOR_CORE_MMA=1
#endif
}

}  // namespace arch
}  // namespace cutlass
