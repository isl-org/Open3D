// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/core/linalg/LinalgUtils.h"

#include "open3d/core/CUDAUtils.h"

namespace open3d {
namespace core {

CuSolverContext& CuSolverContext::GetInstance() {
    static CuSolverContext instance;
    return instance;
}

CuSolverContext::CuSolverContext() {
    for (const Device& device : Device::GetAvailableCUDADevices()) {
        CUDAScopedDevice scoped_device(device);
        cusolverDnHandle_t handle;
        if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS) {
            utility::LogError("Unable to create cuSolver handle for {}.",
                              device.ToString());
        }
        map_device_to_handle_[device] = handle;
    }
}

CuSolverContext::~CuSolverContext() {
    // Destroy map_device_to_handle_
    for (auto& item : map_device_to_handle_) {
        if (cusolverDnDestroy(item.second) != CUSOLVER_STATUS_SUCCESS) {
            utility::LogError(
                    "Unable to destroy cuSolver handle for device {}.",
                    item.first.ToString());
        }
    }
}

cusolverDnHandle_t& CuSolverContext::GetHandle(const Device& device) {
    if (device.GetType() != Device::DeviceType::CUDA) {
        utility::LogError("cuSolver is only available on CUDA devices");
    }
    if (map_device_to_handle_.count(device) == 0) {
        utility::LogError("cuSolver handle not found for device: {}",
                          device.ToString());
    }
    return map_device_to_handle_.at(device);
}

CuBLASContext& CuBLASContext::GetInstance() {
    static CuBLASContext instance;
    return instance;
}

CuBLASContext::CuBLASContext() {
    for (const Device& device : Device::GetAvailableCUDADevices()) {
        CUDAScopedDevice scoped_device(device);
        cublasHandle_t handle;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            utility::LogError("Unable to create cublas handle for {}.",
                              device.ToString());
        }
        map_device_to_handle_[device] = handle;
    }
}

CuBLASContext::~CuBLASContext() {
    // Destroy map_device_to_handle_
    for (auto& item : map_device_to_handle_) {
        if (cublasDestroy(item.second) != CUBLAS_STATUS_SUCCESS) {
            utility::LogError("Unable to destroy cublas handle for device {}.",
                              item.first.ToString());
        }
    }
}

cublasHandle_t& CuBLASContext::GetHandle(const Device& device) {
    if (device.GetType() != Device::DeviceType::CUDA) {
        utility::LogError("cuBLAS is only available on CUDA devices");
    }
    if (map_device_to_handle_.count(device) == 0) {
        utility::LogError("cuBLAS handle not found for device: {}",
                          device.ToString());
    }
    return map_device_to_handle_.at(device);
}

}  // namespace core
}  // namespace open3d
