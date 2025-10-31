// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/CUDAUtils.h"

#include "open3d/Macro.h"
#include "open3d/utility/Logging.h"

#ifdef BUILD_CUDA_MODULE
#include <unordered_set>

#include "open3d/core/MemoryManager.h"
#endif

namespace open3d {
namespace core {
namespace cuda {

int DeviceCount() {
#ifdef BUILD_CUDA_MODULE
    try {
        int num_devices;
        OPEN3D_CUDA_CHECK(cudaGetDeviceCount(&num_devices));
        return num_devices;
    }
    // This function is also used to detect CUDA support in our Python code.
    // Thus, catch any errors if no GPU is available.
    catch (const std::runtime_error&) {
        return 0;
    }
#else
    return 0;
#endif
}

bool IsAvailable() { return cuda::DeviceCount() > 0; }

void ReleaseCache() {
#ifdef BUILD_CUDA_MODULE
#ifdef ENABLE_CACHED_CUDA_MANAGER
    // Release cache from all devices. Since only memory from MemoryManagerCUDA
    // is cached at the moment, this works as expected. In the future, the logic
    // could become more fine-grained.
    MemoryManagerCached::ReleaseCache();
#else
    utility::LogWarning(
            "Built without cached CUDA memory manager, cuda::ReleaseCache() "
            "has no effect.");
#endif

#else
    utility::LogWarning("Built without CUDA module, cuda::ReleaseCache().");
#endif
}

void Synchronize() {
#ifdef BUILD_CUDA_MODULE
    for (int i = 0; i < DeviceCount(); ++i) {
        Synchronize(Device(Device::DeviceType::CUDA, i));
    }
#endif
}

void Synchronize(const Device& device) {
#ifdef BUILD_CUDA_MODULE
    if (device.IsCUDA()) {
        CUDAScopedDevice scoped_device(device);
        OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif
}

void AssertCUDADeviceAvailable(int device_id) {
#ifdef BUILD_CUDA_MODULE
    int num_devices = cuda::DeviceCount();
    if (num_devices == 0) {
        utility::LogError(
                "Invalid device 'CUDA:{}'. -DBUILD_CUDA_MODULE=ON, but no "
                "CUDA device available.",
                device_id);
    } else if (num_devices == 1 && device_id != 0) {
        utility::LogError(
                "Invalid CUDA Device 'CUDA:{}'. Device ID expected to "
                "be 0, but got {}.",
                device_id, device_id);
    } else if (device_id < 0 || device_id >= num_devices) {
        utility::LogError(
                "Invalid CUDA Device 'CUDA:{}'. Device ID expected to "
                "be between 0 to {}, but got {}.",
                device_id, num_devices - 1, device_id);
    }
#else
    utility::LogError(
            "-DBUILD_CUDA_MODULE=OFF. Please build with -DBUILD_CUDA_MODULE=ON "
            "to use CUDA device.");
#endif
}

void AssertCUDADeviceAvailable(const Device& device) {
    if (device.IsCUDA()) {
        AssertCUDADeviceAvailable(device.GetID());
    } else {
        utility::LogError(
                "Expected device-type to be CUDA, but got device '{}'",
                device.ToString());
    }
}

bool SupportsMemoryPools(const Device& device) {
#if defined(BUILD_CUDA_MODULE) && (CUDART_VERSION >= 11020)
    if (device.IsCUDA()) {
        int driverVersion = 0;
        int deviceSupportsMemoryPools = 0;
        OPEN3D_CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
        if (driverVersion >=
            11020) {  // avoid invalid value error in cudaDeviceGetAttribute
            OPEN3D_CUDA_CHECK(cudaDeviceGetAttribute(
                    &deviceSupportsMemoryPools, cudaDevAttrMemoryPoolsSupported,
                    device.GetID()));
        }
        return !!deviceSupportsMemoryPools;
    } else {
        return false;
    }
#else
    return false;
#endif
}

#ifdef BUILD_CUDA_MODULE
int GetDevice() {
    int device;
    OPEN3D_CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

static void SetDevice(int device_id) {
    AssertCUDADeviceAvailable(device_id);
    OPEN3D_CUDA_CHECK(cudaSetDevice(device_id));
}

void Synchronize(const CUDAStream& stream) {
    OPEN3D_CUDA_CHECK(cudaStreamSynchronize(stream.Get()));
}

#endif

}  // namespace cuda

#ifdef BUILD_CUDA_MODULE

CUDAStream& CUDAStream::GetInstance() {
    // The global stream state is given per thread like CUDA's internal
    // device state.
    thread_local CUDAStream instance = CUDAStream::Default();
    return instance;
}

CUDAStream CUDAStream::CreateNew() {
    CUDAStream stream;
    OPEN3D_CUDA_CHECK(cudaStreamCreate(&stream.stream_));
    return stream;
}

void CUDAStream::SetHostToDeviceMemcpyPolicy(CUDAMemoryCopyPolicy policy) {
    OPEN3D_ASSERT(!IsDefaultStream());
    memcpy_from_host_to_device_ = policy;
}

CUDAMemoryCopyPolicy CUDAStream::GetHostToDeviceMemcpyPolicy() const {
    return memcpy_from_host_to_device_;
}

CUDAMemoryCopyPolicy CUDAStream::GetDeviceToHostMemcpyPolicy() const {
    return memcpy_from_device_to_host_;
}

void CUDAStream::SetDeviceToHostMemcpyPolicy(CUDAMemoryCopyPolicy policy) {
    OPEN3D_ASSERT(!IsDefaultStream());
    memcpy_from_device_to_host_ = policy;
}

bool CUDAStream::IsDefaultStream() const {
    return stream_ == static_cast<cudaStream_t>(nullptr);
}

cudaStream_t CUDAStream::Get() const { return stream_; }

void CUDAStream::Set(cudaStream_t stream) { stream_ = stream; }

void CUDAStream::Destroy() {
    OPEN3D_ASSERT(!IsDefaultStream());
    OPEN3D_CUDA_CHECK(cudaStreamDestroy(stream_));
    *this = CUDAStream::Default();
}

CUDAScopedDevice::CUDAScopedDevice(int device_id)
    : prev_device_id_(cuda::GetDevice()) {
    cuda::SetDevice(device_id);
}

CUDAScopedDevice::CUDAScopedDevice(const Device& device)
    : CUDAScopedDevice(device.GetID()) {
    cuda::AssertCUDADeviceAvailable(device);
}

CUDAScopedDevice::~CUDAScopedDevice() { cuda::SetDevice(prev_device_id_); }

CUDAScopedStream::CUDAScopedStream(CUDAStream stream, bool destroy_on_exit)
    : prev_stream_(CUDAStream::GetInstance()),
      new_stream_(stream),
      owns_new_stream_(destroy_on_exit) {
    CUDAStream::GetInstance() = new_stream_;
}

CUDAScopedStream::~CUDAScopedStream() {
    if (owns_new_stream_) {
        OPEN3D_ASSERT((prev_stream_.Get() != new_stream_.Get()) &&
                      "CUDAScopedStream destroy_on_exit would destroy the same "
                      "stream which was in place before the scoped stream was "
                      "created.");
        new_stream_.Destroy();
    }
    CUDAStream::GetInstance() = prev_stream_;
}

CUDAState& CUDAState::GetInstance() {
    static CUDAState instance;
    return instance;
}

bool CUDAState::IsP2PEnabled(int src_id, int tar_id) const {
    cuda::AssertCUDADeviceAvailable(src_id);
    cuda::AssertCUDADeviceAvailable(tar_id);
    return p2p_enabled_[src_id][tar_id];
}

bool CUDAState::IsP2PEnabled(const Device& src, const Device& tar) const {
    cuda::AssertCUDADeviceAvailable(src);
    cuda::AssertCUDADeviceAvailable(tar);
    return p2p_enabled_[src.GetID()][tar.GetID()];
}

void CUDAState::ForceDisableP2PForTesting() {
    for (int src_id = 0; src_id < cuda::DeviceCount(); ++src_id) {
        for (int tar_id = 0; tar_id < cuda::DeviceCount(); ++tar_id) {
            if (src_id != tar_id && p2p_enabled_[src_id][tar_id]) {
                p2p_enabled_[src_id][tar_id] = false;
            }
        }
    }
}

CUDAState::CUDAState() {
    // Check and enable all possible peer to peer access.
    p2p_enabled_ = std::vector<std::vector<bool>>(
            cuda::DeviceCount(), std::vector<bool>(cuda::DeviceCount(), false));

    for (int src_id = 0; src_id < cuda::DeviceCount(); ++src_id) {
        for (int tar_id = 0; tar_id < cuda::DeviceCount(); ++tar_id) {
            if (src_id == tar_id) {
                p2p_enabled_[src_id][tar_id] = true;
            } else {
                CUDAScopedDevice scoped_device(src_id);

                // Check access.
                int can_access = 0;
                OPEN3D_CUDA_CHECK(
                        cudaDeviceCanAccessPeer(&can_access, src_id, tar_id));
                // Enable access.
                if (can_access) {
                    p2p_enabled_[src_id][tar_id] = true;
                    cudaError_t err = cudaDeviceEnablePeerAccess(tar_id, 0);
                    if (err == cudaErrorPeerAccessAlreadyEnabled) {
                        // Ignore error since P2P is already enabled.
                        cudaGetLastError();
                    } else {
                        OPEN3D_CUDA_CHECK(err);
                    }
                } else {
                    p2p_enabled_[src_id][tar_id] = false;
                }
            }
        }
    }
}

int GetCUDACurrentDeviceTextureAlignment() {
    int value;
    OPEN3D_CUDA_CHECK(cudaDeviceGetAttribute(
            &value, cudaDevAttrTextureAlignment, cuda::GetDevice()));
    return value;
}

int GetCUDACurrentWarpSize() {
    int value;
    OPEN3D_CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize,
                                             cuda::GetDevice()));
    return value;
}

size_t GetCUDACurrentTotalMemSize() {
    size_t free;
    size_t total;
    OPEN3D_CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return total;
}

#endif

}  // namespace core
}  // namespace open3d

#ifdef BUILD_CUDA_MODULE

namespace open3d {
namespace core {

const std::unordered_set<cudaError_t> kProcessEndingErrors = {
        cudaErrorAssert,
        cudaErrorLaunchTimeout,
        cudaErrorHardwareStackError,
        cudaErrorIllegalInstruction,
        cudaErrorMisalignedAddress,
        cudaErrorInvalidAddressSpace,
        cudaErrorInvalidPc,
        cudaErrorTensorMemoryLeak,
        cudaErrorMpsClientTerminated,
        cudaErrorExternalDevice,
        cudaErrorContained,
        cudaErrorIllegalAddress,
        cudaErrorLaunchFailure,
        cudaErrorECCUncorrectable,
        cudaErrorUnknown};

void __OPEN3D_CUDA_CHECK(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        if (kProcessEndingErrors.count(err)) {
            utility::LogError(
                    "{}:{} CUDA runtime error: {}. This is a process-ending "
                    "error. All further operations will fail and the process "
                    "needs to be relaunched to be able to use CUDA.",
                    file, line, cudaGetErrorString(err));
        } else {
            utility::LogError("{}:{} CUDA runtime error: {}", file, line,
                              cudaGetErrorString(err));
        }
    }
}

void __OPEN3D_GET_LAST_CUDA_ERROR(const char* message,
                                  const char* file,
                                  const int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        utility::LogError("{}:{} {}: OPEN3D_GET_LAST_CUDA_ERROR(): {}", file,
                          line, message, cudaGetErrorString(err));
    }
}

}  // namespace core
}  // namespace open3d

#endif

// C interface to provide un-mangled function to Python ctypes
extern "C" OPEN3D_DLL_EXPORT int open3d_core_cuda_device_count() {
    return open3d::core::cuda::DeviceCount();
}
