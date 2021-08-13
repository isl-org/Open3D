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

#include "open3d/core/CUDAUtils.h"

#include "open3d/Macro.h"
#include "open3d/utility/Logging.h"

#ifdef BUILD_CUDA_MODULE
#include "open3d/core/MemoryManager.h"
#endif

namespace open3d {
namespace core {
namespace cuda {

int DeviceCount() {
#ifdef BUILD_CUDA_MODULE
    try {
        std::shared_ptr<CUDAState> cuda_state = CUDAState::GetInstance();
        return cuda_state->GetNumDevices();
    } catch (const std::runtime_error&) {  // GetInstance can throw
        return 0;
    }
#else
    return 0;
#endif
}

bool IsAvailable() { return cuda::DeviceCount() > 0; }

void ReleaseCache() {
#ifdef BUILD_CUDA_MODULE
#ifdef BUILD_CACHED_CUDA_MANAGER
    // Release cache from all devices. Since only memory from CUDAMemoryManager
    // is cached at the moment, this works as expected. In the future, the logic
    // could become more fine-grained.
    CachedMemoryManager::ReleaseCache();
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
        CUDAScopedDevice scoped_device(Device("CUDA", i));
        OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif
}

void Synchronize(const Device& device) {
#ifdef BUILD_CUDA_MODULE
    if (device.GetType() == Device::DeviceType::CUDA) {
        CUDAScopedDevice scoped_device(device);
        OPEN3D_CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif
}

void AssertCUDADeviceAvailable(int device_id) {
    // #ifdef BUILD_CUDA_MODULE
    int num_devices = 0;
    OPEN3D_CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    utility::LogInfo(" ################# NUM DEVICE : {}", num_devices);
    // if (DeviceCount() == 0) {
    //     utility::LogError(
    //             "Invalid device 'CUDA:{}'. -DBUILD_CUDA_MODULE=ON, but no "
    //             "CUDA device available.",
    //             device_id);
    // }
    //  else if (DeviceCount() == 1 && device_id != 0) {
    //     utility::LogError(
    //             "Invalid CUDA Device 'CUDA:{}'. Device ID expected to "
    //             "be 0, but got {}.",
    //             device_id, device_id);
    // } else if (device_id < 0 || device_id >= DeviceCount()) {
    //     utility::LogError(
    //             "Invalid CUDA Device 'CUDA:{}'. Device ID expected to "
    //             "be between 0 to {}, but got {}.",
    //             device_id, DeviceCount() - 1, device_id);
    // }
    // #else
    //     utility::LogError(
    //             "-DBUILD_CUDA_MODULE=OFF. Please build with
    //             -DBUILD_CUDA_MODULE=ON " "to use CUDA device.");
    // #endif
}

void AssertCUDADeviceAvailable(const Device& device) {
    if (device.GetType() == Device::DeviceType::CUDA) {
        AssertCUDADeviceAvailable(device.GetID());
    } else {
        utility::LogError(
                "Expected device-type to be CUDA, but got device '{}'",
                device.ToString());
    }
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

class CUDAStream {
public:
    static CUDAStream& GetInstance() {
        // The global stream state is given per thread like CUDA's internal
        // device state.
        static thread_local CUDAStream instance;
        return instance;
    }

    cudaStream_t Get() { return stream_; }
    void Set(cudaStream_t stream) { stream_ = stream; }

    static cudaStream_t Default() { return static_cast<cudaStream_t>(0); }

private:
    CUDAStream() = default;

    cudaStream_t stream_ = Default();
};

cudaStream_t GetStream() { return CUDAStream::GetInstance().Get(); }

static void SetStream(cudaStream_t stream) {
    CUDAStream::GetInstance().Set(stream);
}

cudaStream_t GetDefaultStream() { return CUDAStream::Default(); }

#endif

}  // namespace cuda

#ifdef BUILD_CUDA_MODULE

CUDAScopedDevice::CUDAScopedDevice(int device_id)
    : prev_device_id_(cuda::GetDevice()) {
    cuda::SetDevice(device_id);
}

CUDAScopedDevice::CUDAScopedDevice(const Device& device)
    : CUDAScopedDevice(device.GetID()) {}

CUDAScopedDevice::~CUDAScopedDevice() { cuda::SetDevice(prev_device_id_); }

constexpr CUDAScopedStream::CreateNewStreamTag
        CUDAScopedStream::CreateNewStream;

CUDAScopedStream::CUDAScopedStream(const CreateNewStreamTag&)
    : prev_stream_(cuda::GetStream()), owns_new_stream_(true) {
    OPEN3D_CUDA_CHECK(cudaStreamCreate(&new_stream_));
    cuda::SetStream(new_stream_);
}

CUDAScopedStream::CUDAScopedStream(cudaStream_t stream)
    : prev_stream_(cuda::GetStream()),
      new_stream_(stream),
      owns_new_stream_(false) {
    cuda::SetStream(stream);
}

CUDAScopedStream::~CUDAScopedStream() {
    if (owns_new_stream_) {
        OPEN3D_CUDA_CHECK(cudaStreamDestroy(new_stream_));
    }
    cuda::SetStream(prev_stream_);
}

std::shared_ptr<CUDAState> CUDAState::GetInstance() {
    static std::shared_ptr<CUDAState> instance{new CUDAState};
    return instance;
}

bool CUDAState::IsP2PEnabled(int src_id, int tar_id) {
    if (src_id < 0 || src_id >= num_devices_) {
        utility::LogError("Device id {} is out of bound of total {} devices.",
                          src_id, num_devices_);
    }
    if (tar_id < 0 || tar_id >= num_devices_) {
        utility::LogError("Device id {} is out of bound of total {} devices.",
                          tar_id, num_devices_);
    }
    return p2p_enabled_[src_id][tar_id];
}

bool CUDAState::IsP2PEnabled(const Device& src, const Device& tar) {
    return p2p_enabled_[src.GetID()][tar.GetID()];
}

std::vector<std::vector<bool>> CUDAState::GetP2PEnabled() const {
    return p2p_enabled_;
}

int CUDAState::GetNumDevices() const { return num_devices_; }

int CUDAState::GetWarpSize() const { return warp_sizes_[GetCurrentDeviceID()]; }

int CUDAState::GetCurrentDeviceID() const { return cuda::GetDevice(); }

/// Disable P2P device transfer by marking p2p_enabled_ to `false`, in order
/// to run non-p2p tests on a p2p-capable machine.
void CUDAState::ForceDisableP2PForTesting() {
    for (int src_id = 0; src_id < num_devices_; ++src_id) {
        for (int tar_id = 0; tar_id < num_devices_; ++tar_id) {
            if (src_id != tar_id && p2p_enabled_[src_id][tar_id]) {
                p2p_enabled_[src_id][tar_id] = false;
            }
        }
    }
}

CUDAState::CUDAState() {
    OPEN3D_CUDA_CHECK(cudaGetDeviceCount(&num_devices_));

    // Check and enable all possible peer to peer access.
    p2p_enabled_ = std::vector<std::vector<bool>>(
            num_devices_, std::vector<bool>(num_devices_, false));

    for (int src_id = 0; src_id < num_devices_; ++src_id) {
        for (int tar_id = 0; tar_id < num_devices_; ++tar_id) {
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
                        // Ignore error since p2p is already enabled.
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

    // Cache warp sizes
    warp_sizes_.resize(num_devices_);
    for (int device_id = 0; device_id < num_devices_; ++device_id) {
        cudaDeviceProp device_prop;
        OPEN3D_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
        warp_sizes_[device_id] = device_prop.warpSize;
    }
}

int GetCUDACurrentDeviceTextureAlignment() {
    int value = 0;
    cudaError_t err = cudaDeviceGetAttribute(
            &value, cudaDevAttrTextureAlignment, cuda::GetDevice());
    if (err != cudaSuccess) {
        utility::LogError(
                "GetCUDACurrentDeviceTextureAlignment(): "
                "cudaDeviceGetAttribute failed with {}",
                cudaGetErrorString(err));
    }
    return value;
}

#endif

}  // namespace core
}  // namespace open3d

#ifdef BUILD_CUDA_MODULE
namespace open3d {
namespace core {

void __OPEN3D_CUDA_CHECK(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        utility::LogError("{}:{} CUDA runtime error: {}", file, line,
                          cudaGetErrorString(err));
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
