// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

/// \file CUDAState.cuh
///
/// CUDAState.cuh can only be included by nvcc compiled source code.

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

#include "Open3D/Core/CUDAUtils.h"
#include "Open3D/Core/Device.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

/// \class CUDADeviceSwitcher
///
/// Switch CUDA device id in the current scope. The device id will be resetted
/// once leaving the scope.
///
/// CUDADeviceSwitcher provies an
/// [RAII-style](https://en.wikipedia.org/wiki/Resource_acquisition_is_initialization)
/// mechanism for setting and resetting CUDA devices of a scoped block.
///
/// Example:
/// ```cpp
/// void my_func() {
///     // The switcher recoreds the previous device when it is constructed.
///     // Let's assume cudaGetDevice == 0 initially.
///     CUDADeviceSwitcher switcher;
///
///     switcher.SwitchTo(1);
///     // Now cudaGetDevice == 1.
///     // Make cuda calls here for device 1.
///
///     switcher.SwitchTo(2);
///     // Now cudaGetDevice == 2.
///     // Make cuda calls here for device 1.
///
///     // After `my_func` returns, `switcher` goes out-of-scope,
///     // so cudaGetDevice will be reset back to 0.
/// }
/// ```
///
/// You may also directly initialize and switch to a device:
/// void my_func() {
///     // The switcher recoreds the previous device and switch to device 1.
///     CUDADeviceSwitcher switcher(1);
///
///     // After `my_func` returns, `switcher` goes out-of-scope,
///     // so cudaGetDevice will be reset back to the previous device.
/// }
class CUDADeviceSwitcher {
public:
    /// Init CUDADeviceSwitcher class and keep using the current device.
    CUDADeviceSwitcher() { OPEN3D_CUDA_CHECK(cudaGetDevice(&prev_device_id_)); }

    CUDADeviceSwitcher(int device_id) : CUDADeviceSwitcher() {
        SwitchTo(device_id);
    }

    CUDADeviceSwitcher(const Device& device)
        : CUDADeviceSwitcher(device.GetID()) {}

    void SwitchTo(int device_id) const {
        OPEN3D_CUDA_CHECK(cudaSetDevice(device_id));
    }

    void SwitchTo(const Device& device) const { SwitchTo(device.GetID()); }

    ~CUDADeviceSwitcher() { OPEN3D_CUDA_CHECK(cudaSetDevice(prev_device_id_)); }

    CUDADeviceSwitcher(CUDADeviceSwitcher const&) = delete;

    void operator=(CUDADeviceSwitcher const&) = delete;

private:
    int prev_device_id_;
};

/// CUDAState is a lazy-evaluated singleton class that initializes and stores
/// the states of CUDA devices.
///
/// Currenty is stores total number of devices and peer-to-peer availbility.
///
/// In the future, it can also be used to store
/// - Device allocators
/// - cuBLAS and cuSPARSE handles
/// - Scratch space sizes
/// - ...
///
/// Ref:
/// https://stackoverflow.com/a/1008289/1255535
/// https://stackoverflow.com/a/40337728/1255535
/// https://github.com/pytorch/pytorch/blob/master/aten/src/THC/THCGeneral.cpp
class CUDAState {
public:
    static std::shared_ptr<CUDAState> GetInstance() {
        static std::shared_ptr<CUDAState> instance{new CUDAState};
        return instance;
    }

    ~CUDAState() {}

    CUDAState(CUDAState const&) = delete;

    void operator=(CUDAState const&) = delete;

    bool IsP2PEnabled(int src_id, int tar_id) {
        if (src_id < 0 || src_id >= num_devices_) {
            utility::LogError(
                    "Device id {} is out of bound of total {} devices.", src_id,
                    num_devices_);
        }
        if (tar_id < 0 || tar_id >= num_devices_) {
            utility::LogError(
                    "Device id {} is out of bound of total {} devices.", tar_id,
                    num_devices_);
        }
        return p2p_enabled_[src_id][tar_id];
    }

    bool IsP2PEnabled(const Device& src, const Device& tar) {
        return p2p_enabled_[src.GetID()][tar.GetID()];
    }

    std::vector<std::vector<bool>> GetP2PEnabled() const {
        return p2p_enabled_;
    }

    int GetNumDevices() const { return num_devices_; }

    int GetWarpSize() const { return warp_sizes_[GetCurentDeviceID()]; }

    int GetCurentDeviceID() const {
        int device_id;
        OPEN3D_CUDA_CHECK(cudaGetDevice(&device_id));
        return device_id;
    }

    /// Disable P2P device transfer by marking p2p_enabled_ to `false`, in order
    /// to run non-p2p tests on a p2p-capable machine.
    void ForceDisableP2PForTesting() {
        CUDADeviceSwitcher switcher;
        for (int src_id = 0; src_id < num_devices_; ++src_id) {
            for (int tar_id = 0; tar_id < num_devices_; ++tar_id) {
                if (src_id != tar_id && p2p_enabled_[src_id][tar_id]) {
                    p2p_enabled_[src_id][tar_id] = false;
                }
            }
        }
    }

private:
    CUDAState() {
        CUDADeviceSwitcher switcher;
        OPEN3D_CUDA_CHECK(cudaGetDeviceCount(&num_devices_));

        // Check and enable all possible peer to peer access.
        p2p_enabled_ = std::vector<std::vector<bool>>(
                num_devices_, std::vector<bool>(num_devices_, false));

        for (int src_id = 0; src_id < num_devices_; ++src_id) {
            for (int tar_id = 0; tar_id < num_devices_; ++tar_id) {
                if (src_id == tar_id) {
                    p2p_enabled_[src_id][tar_id] = true;
                } else {
                    switcher.SwitchTo(src_id);
                    // Check access.
                    int can_access = 0;
                    OPEN3D_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access,
                                                              src_id, tar_id));
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

private:
    int num_devices_ = 0;
    std::vector<int> warp_sizes_;
    std::vector<std::vector<bool>> p2p_enabled_;
};

}  // namespace open3d
