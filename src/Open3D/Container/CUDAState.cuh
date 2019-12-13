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

#include "Open3D/Container/CUDAUtils.h"
#include "Open3D/Container/Device.h"
#include "Open3D/Utility/Console.h"

namespace open3d {

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

private:
    CUDAState() {
        // Cache current device for recovery.
        int prev_dev = 0;
        OPEN3D_CUDA_CHECK(cudaGetDevice(&prev_dev));

        // Get number of devices.
        OPEN3D_CUDA_CHECK(cudaGetDeviceCount(&num_devices_));

        // Check and enable all possible peer to peer access.
        p2p_enabled_ = std::vector<std::vector<bool>>(
                num_devices_, std::vector<bool>(num_devices_, false));

        // Note: To run non-p2p tests on a p2p capable machine, uncomment the
        // following lines and comment out the lines where p2p is enabled.
        //
        // for (int src_id = 0; src_id < num_devices_; ++src_id) {
        //     for (int tar_id = 0; tar_id < num_devices_; ++tar_id) {
        //         if (src_id == tar_id) {
        //             p2p_enabled_[src_id][tar_id] = true;
        //         }
        //     }
        // }

        for (int src_id = 0; src_id < num_devices_; ++src_id) {
            for (int tar_id = 0; tar_id < num_devices_; ++tar_id) {
                if (src_id == tar_id) {
                    p2p_enabled_[src_id][tar_id] = true;
                } else {
                    OPEN3D_CUDA_CHECK(cudaSetDevice(src_id));
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

        // Restore previous device.
        OPEN3D_CUDA_CHECK(cudaSetDevice(prev_dev));
    }

private:
    int num_devices_;
    std::vector<std::vector<bool>> p2p_enabled_;
};

/// Switch CUDA device id in the current scope. The device id will be resetted
/// once leaving the scope.
class CUDASwitchDevice {
public:
    CUDASwitchDevice(int device_id) {
        OPEN3D_CUDA_CHECK(cudaGetDevice(&prev_device_id_));
        if (device_id != prev_device_id_) {
            OPEN3D_CUDA_CHECK(cudaSetDevice(device_id));
        }
    }

    CUDASwitchDevice(const Device& device) : CUDASwitchDevice(device.GetID()) {}

    void SwitchTo(int device_id) const {
        OPEN3D_CUDA_CHECK(cudaSetDevice(device_id));
    }

    void SwitchTo(const Device& device) const { SwitchTo(device.GetID()); }

    ~CUDASwitchDevice() { OPEN3D_CUDA_CHECK(cudaSetDevice(prev_device_id_)); }

private:
    int prev_device_id_;
};

}  // namespace open3d
