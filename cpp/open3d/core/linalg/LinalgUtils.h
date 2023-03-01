// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "open3d/core/Device.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/MemoryManager.h"
#include "open3d/core/linalg/LinalgHeadersCPU.h"
#include "open3d/core/linalg/LinalgHeadersCUDA.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

#define DISPATCH_LINALG_DTYPE_TO_TEMPLATE(DTYPE, ...)    \
    [&] {                                                \
        if (DTYPE == open3d::core::Float32) {            \
            using scalar_t = float;                      \
            return __VA_ARGS__();                        \
        } else if (DTYPE == open3d::core::Float64) {     \
            using scalar_t = double;                     \
            return __VA_ARGS__();                        \
        } else {                                         \
            utility::LogError("Unsupported data type."); \
        }                                                \
    }()

inline void OPEN3D_LAPACK_CHECK(OPEN3D_CPU_LINALG_INT info,
                                const std::string& msg) {
    if (info < 0) {
        utility::LogError("{}: {}-th parameter is invalid.", msg, -info);
    } else if (info > 0) {
        utility::LogError("{}: singular condition detected.", msg);
    }
}

#ifdef BUILD_CUDA_MODULE
inline void OPEN3D_CUBLAS_CHECK(cublasStatus_t status, const std::string& msg) {
    if (CUBLAS_STATUS_SUCCESS != status) {
        utility::LogError("{}", msg);
    }
}

inline void OPEN3D_CUSOLVER_CHECK(cusolverStatus_t status,
                                  const std::string& msg) {
    if (CUSOLVER_STATUS_SUCCESS != status) {
        utility::LogError("{}", msg);
    }
}

inline void OPEN3D_CUSOLVER_CHECK_WITH_DINFO(cusolverStatus_t status,
                                             const std::string& msg,
                                             int* dinfo,
                                             const Device& device) {
    int hinfo;
    MemoryManager::MemcpyToHost(&hinfo, dinfo, device, sizeof(int));
    if (status != CUSOLVER_STATUS_SUCCESS || hinfo != 0) {
        if (hinfo < 0) {
            utility::LogError("{}: {}-th parameter is invalid.", msg, -hinfo);
        } else if (hinfo > 0) {
            utility::LogError("{}: singular condition detected.", msg);
        } else {
            utility::LogError("{}: status error code = {}.", msg, status);
        }
    }
}

class CuSolverContext {
public:
    static CuSolverContext& GetInstance();

    CuSolverContext(const CuSolverContext&) = delete;
    CuSolverContext& operator=(const CuSolverContext&) = delete;
    ~CuSolverContext();

    cusolverDnHandle_t& GetHandle(const Device& device);

private:
    CuSolverContext();
    std::unordered_map<Device, cusolverDnHandle_t> map_device_to_handle_;
};

class CuBLASContext {
public:
    static CuBLASContext& GetInstance();

    CuBLASContext(const CuBLASContext&) = delete;
    CuBLASContext& operator=(const CuBLASContext&) = delete;
    ~CuBLASContext();

    cublasHandle_t& GetHandle(const Device& device);

private:
    CuBLASContext();
    std::unordered_map<Device, cublasHandle_t> map_device_to_handle_;
};
#endif
}  // namespace core
}  // namespace open3d
