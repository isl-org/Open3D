// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

/// \file CUDAUtils.h
/// \brief Common CUDA utilities
///
/// CUDAUtils.h may be included from CPU-only code.
/// Use \#ifdef __CUDACC__ to mark conditional compilation

#pragma once

#include "open3d/core/Device.h"
#include "open3d/utility/Logging.h"

#ifdef BUILD_CUDA_MODULE

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "open3d/utility/Optional.h"

#define OPEN3D_FORCE_INLINE __forceinline__
#define OPEN3D_HOST_DEVICE __host__ __device__
#define OPEN3D_DEVICE __device__
#define OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(type)                            \
    static_assert(__nv_is_extended_host_device_lambda_closure_type(type), \
                  #type " must be a __host__ __device__ lambda")
#define OPEN3D_CUDA_CHECK(err) \
    open3d::core::__OPEN3D_CUDA_CHECK(err, __FILE__, __LINE__)
#define OPEN3D_GET_LAST_CUDA_ERROR(message) \
    __OPEN3D_GET_LAST_CUDA_ERROR(message, __FILE__, __LINE__)
#define CUDA_CALL(cuda_function, ...) cuda_function(__VA_ARGS__);

#else  // #ifdef BUILD_CUDA_MODULE

#define OPEN3D_FORCE_INLINE inline
#define OPEN3D_HOST_DEVICE
#define OPEN3D_DEVICE
#define OPEN3D_ASSERT_HOST_DEVICE_LAMBDA(type)
#define OPEN3D_CUDA_CHECK(err)
#define OPEN3D_GET_LAST_CUDA_ERROR(message)
#define CUDA_CALL(cuda_function, ...) \
    utility::LogError("Not built with CUDA, cannot call " #cuda_function);

#endif  // #ifdef BUILD_CUDA_MODULE

namespace open3d {
namespace core {

#ifdef BUILD_CUDA_MODULE

/// \class CUDAScopedDevice
///
/// Switch CUDA device id in the current scope. The device id will be reset
/// once leaving the scope.
///
/// CUDAScopedDevice provides an
/// [RAII-style](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization)
/// mechanism for setting and resetting CUDA devices of a scoped block.
///
/// Example:
/// ```cpp
/// void my_func() {
///     // The scoped wrapper records the previous device when it is
///     constructed.
///     // Let's assume the current device is 0 initially.
///     CUDAScopedDevice scoped_device(1);
///
///     // Now the current device is 1.
///     // Make cuda calls here for device 1.
///
///     // After `my_func` returns, `scoped_device` goes out-of-scope,
///     // so the global device will be reset back to 0.
/// }
class CUDAScopedDevice {
public:
    explicit CUDAScopedDevice(int device_id);

    explicit CUDAScopedDevice(const Device& device);

    ~CUDAScopedDevice();

    CUDAScopedDevice(const CUDAScopedDevice&) = delete;
    CUDAScopedDevice& operator=(const CUDAScopedDevice&) = delete;

private:
    int prev_device_id_;
};

/// \class CUDAScopedStream
///
/// Switch CUDA stream in the current scope. The stream will be reset
/// once leaving the scope.
///
/// CUDAScopedStream provides an
/// [RAII-style](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization)
/// mechanism for setting and resetting CUDA streams of a scoped block.
///
/// Example:
/// ```cpp
/// void my_func(cudaStream_t stream) {
///     // The scoped wrapper records the previous stream when it is
///     constructed.
///     // Let's assume the current stream is 0 initially.
///     CUDAScopedStream scoped_stream(stream);
///
///     // Now the current stream is 1.
///     // Make cuda calls here for stream 1.
///
///     // After `my_func` returns, `scoped_stream` goes out-of-scope,
///     // so the global stream will be reset back to 0.
/// }
/// ```
/// If a new stream shall be used, the following constructor can be used:
/// ```cpp
/// void my_func() {
///     // The scoped wrapper records the previous stream when it is
///     constructed.
///     // Let's assume the current stream is 0 initially.
///     CUDAScopedStream scoped_stream(CUDAScopedStream::CreateNewStream);
///
///     // Now the current stream is 1.
///     // Make cuda calls here for stream 1.
///
///     // After `my_func` returns, `scoped_stream` goes out-of-scope,
///     // so the global stream will be reset back to 0.
/// }
/// ```
class CUDAScopedStream {
private:
    struct CreateNewStreamTag {
        CreateNewStreamTag(const CreateNewStreamTag&) = delete;
        CreateNewStreamTag& operator=(const CreateNewStreamTag&) = delete;
        CreateNewStreamTag(CreateNewStreamTag&&) = delete;
        CreateNewStreamTag& operator=(CreateNewStreamTag&&) = delete;
    };

public:
    constexpr static CreateNewStreamTag CreateNewStream = {};

    explicit CUDAScopedStream(const CreateNewStreamTag&);

    explicit CUDAScopedStream(cudaStream_t stream);

    ~CUDAScopedStream();

    CUDAScopedStream(const CUDAScopedStream&) = delete;
    CUDAScopedStream& operator=(const CUDAScopedStream&) = delete;

private:
    cudaStream_t prev_stream_;
    cudaStream_t new_stream_;
    bool owns_new_stream_ = false;
};

/// CUDAState is a lazy-evaluated singleton class that initializes and stores
/// the states of CUDA devices.
///
/// Currently it stores the peer-to-peer availability.
///
/// In the future, it can also be used to store
/// - Device allocators
/// - cuBLAS and cuSPARSE handles
/// - Scratch space sizes
/// - ...
///
/// Ref:
/// https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCGeneral.cpp
class CUDAState {
public:
    static CUDAState& GetInstance();

    CUDAState(const CUDAState&) = delete;
    CUDAState& operator=(const CUDAState&) = delete;

    /// Returns true if peer-to-peer is available from the CUDA device-ID
    /// \p src_id to \p tar_id.
    bool IsP2PEnabled(int src_id, int tar_id) const;

    /// Returns true if peer-to-peer is available from the CUDA device
    /// \p src to \p tar.
    bool IsP2PEnabled(const Device& src, const Device& tar) const;

    /// Disables P2P device transfer between all devices, in order to run
    /// non-P2P tests on a P2P-capable machine.
    void ForceDisableP2PForTesting();

private:
    CUDAState();

    std::vector<std::vector<bool>> p2p_enabled_;
};

/// Returns the size of a warp for the current device.
int GetCUDACurrentWarpSize();

/// Returns the texture alignment in bytes for the current device.
int GetCUDACurrentDeviceTextureAlignment();

/// Returns the size of total global memory for the current device.
size_t GetCUDACurrentTotalMemSize();
#endif

namespace cuda {

/// Returns the number of available CUDA devices. Returns 0 if Open3D is not
/// compiled with CUDA support.
int DeviceCount();

/// Returns true if Open3D is compiled with CUDA support and at least one
/// compatible CUDA device is detected.
bool IsAvailable();

/// Releases CUDA memory manager cache. This is typically used for debugging.
void ReleaseCache();

/// Calls cudaDeviceSynchronize() for all CUDA devices. If Open3D is not
/// compiled with CUDA this function has no effect.
void Synchronize();

/// Calls cudaDeviceSynchronize() for the specified device. If Open3D is not
/// compiled with CUDA or if \p device is not a CUDA device, this function has
/// no effect.
/// \param device The device to be synchronized.
void Synchronize(const Device& device);

/// Checks if the CUDA device-ID is available and throws error if not. The CUDA
/// device-ID must be between 0 to device count - 1.
/// \param device_id The cuda device id to be checked.
void AssertCUDADeviceAvailable(int device_id);

/// Checks if the CUDA device-ID is available and throws error if not. The CUDA
/// device-ID must be between 0 to device count - 1.
/// \param device The device to be checked.
void AssertCUDADeviceAvailable(const Device& device);

#ifdef BUILD_CUDA_MODULE

int GetDevice();
cudaStream_t GetStream();
cudaStream_t GetDefaultStream();

#endif

}  // namespace cuda
}  // namespace core
}  // namespace open3d

// Exposed as implementation detail of macros at the end of the file.
#ifdef BUILD_CUDA_MODULE

namespace open3d {
namespace core {

void __OPEN3D_CUDA_CHECK(cudaError_t err, const char* file, const int line);

void __OPEN3D_GET_LAST_CUDA_ERROR(const char* message,
                                  const char* file,
                                  const int line);

}  // namespace core
}  // namespace open3d

#endif
