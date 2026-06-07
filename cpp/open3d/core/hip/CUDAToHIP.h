// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// CUDA-to-HIP compatibility shim for the ROCm/HIP build (USE_HIP=ON).
//
// This is the single file that knows about HIP. It is force-included on every
// HIP translation unit (cmake/Open3DSetGlobalProperties.cmake adds it via the
// HIP language -include flag) so the CUDA-spelled core sources compile under
// hipcc unchanged. On the NVIDIA build this header is never referenced and the
// CUDA path is byte-for-byte unaffected.
//
// libc host declarations must win over HIP's __device__ memcpy/memset
// overloads, so pull them in before <hip/hip_runtime.h>.

#pragma once

#if defined(USE_HIP)

#include <cstdlib>
#include <cstring>

#include <hip/hip_runtime.h>

// Open3D gates device-vs-host code throughout core/t on __CUDACC__ (device
// decorations, kernel launch, host fallbacks). We do NOT fake __CUDACC__ for
// HIP: rocThrust's HIP backend disables its for_each/copy specializations when
// __CUDACC__ is set (it reads __CUDACC__ as "use the CUDA backend", which
// contradicts THRUST_DEVICE_SYSTEM=HIP and dumps every thrust call into the
// "unimplemented for this system" generic fallback). Instead the Open3D device
// guards are extended to also accept __HIPCC__ at each site. Provide
// OPEN3D_DEVICE_CODE as the device-vs-host
// discriminator: __CUDA_ARCH__ on CUDA, __HIP_DEVICE_COMPILE__ on HIP.
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#define OPEN3D_DEVICE_CODE 1
#else
#define OPEN3D_DEVICE_CODE 0
#endif
// Eigen would take its CUDA path (<math_constants.h>, absent on ROCm) under a
// CUDA compiler; EIGEN_NO_CUDA keeps it on its native HIP path. Harmless when
// no device TU includes Eigen.
#if !defined(EIGEN_NO_CUDA)
#define EIGEN_NO_CUDA
#endif

// hipCUB lives under the hipcub:: namespace and <hipcub/...> headers; the core
// sources spell it cub::/<cub/cub.cuh>. The forwarding header hip_compat/cub
// remaps the include; this remaps the namespace token.
#define cub hipcub

// Full-wavefront lane mask for the *_sync warp primitives. HIP's
// __shfl_*_sync / __ballot_sync / __any_sync take a 64-bit mask and
// static_assert sizeof(mask)==8, so the upstream 0xffffffff (32-bit) literal
// does not compile. ~0ull is the all-lanes mask on both wave32 and wave64.
#define OPEN3D_FULL_WARP_MASK (~0ull)

// HIP declares __ffs(int) and __ffs(unsigned int) but no 64-bit overload, so
// __ffs(__ballot_sync(...)) -- whose ballot result is unsigned long long on a
// 64-lane wavefront (SlabHash work-queue loop) -- is AMBIGUOUS (the ull arg
// converts to int and unsigned equally). Add the exact 64-bit overload as
// __host__ __device__ so it resolves in both passes; __ffsll is device-only,
// so fall back to __builtin_ffsll on the host pass.
__host__ __device__ __forceinline__ int __ffs(unsigned long long x) {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
    return __ffsll(x);
#else
    return __builtin_ffsll(static_cast<long long>(x));
#endif
}

// CUDA runtime symbols used by the core GPU sources, mapped to their HIP
// equivalents. Only the symbols the build actually references are listed.
#define CUDA_VERSION 9000  // gate the >=9000 *_sync paths in the FAISS sources

#define cudaError_t hipError_t
#define cudaError hipError
#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError

#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamQuery hipStreamQuery
#define cudaStreamGetFlags hipStreamGetFlags

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaMallocManaged hipMallocManaged
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemGetInfo hipMemGetInfo

#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion

#define cudaDevAttrWarpSize hipDeviceAttributeWarpSize
#define cudaDevAttrTextureAlignment hipDeviceAttributeTextureAlignment
#define cudaDevAttrMemoryPoolsSupported hipDeviceAttributeMemoryPoolsSupported
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor

#define cudaPointerAttributes hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes

#endif  // USE_HIP
