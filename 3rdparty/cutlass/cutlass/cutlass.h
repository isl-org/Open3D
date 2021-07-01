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
    \brief Basic include for CUTLASS macros
*/

#pragma once


////////////////////////////////////////////////////////////////////////////////////////////////////

#define CUTLASS_MAJOR 1
#define CUTLASS_MINOR 3
#define CUTLASS_PATCH 2
#define CUTLASS_VERSION ((CUTLASS_MAJOR)*100 + (CUTLASS_MINOR)*10 + CUTLASS_PATCH)

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE __forceinline__ __device__
#elif defined(__CUDACC_RTC__)
#define CUTLASS_HOST_DEVICE __forceinline__ __device__
#define CUTLASS_DEVICE __forceinline__ __device__
#else
#define CUTLASS_HOST_DEVICE
// CUTLASS_DEVICE is an error if not compiling device code
#endif

// CUDA 10.1 introduces the mma instruction
#if !defined(CUTLASS_ENABLE_TENSOR_CORE_MMA)
#define CUTLASS_ENABLE_TENSOR_CORE_MMA 0
#endif

// CUTLASS assert
#define CUTLASS_ASSERT(x) assert(x)

// CUTLASS_PRAGMA_(UNROLL|NO_UNROLL) optimization directives for the CUDA compiler.
#if defined(__CUDA_ARCH__)
#ifdef __NVCC__
    #define CUTLASS_PRAGMA_UNROLL #pragma unroll
    #define CUTLASS_PRAGMA_NO_UNROLL #pragma unroll 1
#elif defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
    #define CUTLASS_PRAGMA_UNROLL _Pragma("unroll")
    #define CUTLASS_PRAGMA_NO_UNROLL _Pragma("unroll 1")
#endif

    #define CUTLASS_GEMM_LOOP CUTLASS_PRAGMA_NO_UNROLL

        #define CUTLASS_GEMM_LOOP_HEADER                \
            asm volatile (".pragma \"nounroll\";\n");
#else

    #define CUTLASS_PRAGMA_UNROLL
    #define CUTLASS_PRAGMA_NO_UNROLL
    #define CUTLASS_GEMM_LOOP_HEADER
    #define CUTLASS_GEMM_LOOP

#endif

// A small helper class to dump a type at compile time
// Usage:: DumpType<Class>::Class
template <typename T>
struct DebugType {};

template <typename T>
CUTLASS_HOST_DEVICE
void DebugTypeFunc(T const& t) {
  T::t;
}

// A small helper class to dump a compile time constant at compile time
// Usage: DumpValue<Class::kConstant>::kConstant
template <int Value>
struct DebugValue {};

namespace cutlass {

/// NVIDIA GPU Warp size
static const int kWarpSize = 32;

}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////////////////////////
