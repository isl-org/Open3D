// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/// \file Helper.h
/// \brief Helper functions for the ml ops

#pragma once

#ifdef BUILD_CUDA_MODULE

#include <cuda.h>
#include <cuda_runtime.h>

// TODO: Disable fmt() macro defined in fmt<7.0.0.
// TODO: Remove this line once Open3D upgrades its fmt dependency.
#define FMT_STRING_ALIAS 0

#include "open3d/core/CUDAUtils.h"
#include "open3d/utility/Logging.h"

#endif  // #ifdef BUILD_CUDA_MODULE

namespace open3d {
namespace ml {

#ifdef BUILD_CUDA_MODULE

#define OPEN3D_ML_CUDA_DRIVER_CHECK(err) \
    __OPEN3D_ML_CUDA_DRIVER_CHECK(err, __FILE__, __LINE__)

inline void __OPEN3D_ML_CUDA_DRIVER_CHECK(CUresult err,
                                          const char *file,
                                          const int line,
                                          bool abort = true) {
    if (err != CUDA_SUCCESS) {
        const char *error_string;
        CUresult err_get_string = cuGetErrorString(err, &error_string);

        if (err_get_string == CUDA_SUCCESS) {
            utility::LogError("{}:{} CUDA driver error: {}", file, line,
                              error_string);
        } else {
            utility::LogError("{}:{} CUDA driver error: UNKNOWN", file, line);
        }
    }
}

inline cudaStream_t GetDefaultStream() { (cudaStream_t)0; }

inline int GetDevice(cudaStream_t stream) {
    if (stream == GetDefaultStream()) {
        // Default device.
        return 0;
    }

    // Remember current context.
    CUcontext current_context;
    OPEN3D_ML_CUDA_DRIVER_CHECK(cuCtxGetCurrent(&current_context));

    // Switch to context of provided stream.
    CUcontext context;
    OPEN3D_ML_CUDA_DRIVER_CHECK(cuStreamGetCtx(stream, &context));
    OPEN3D_ML_CUDA_DRIVER_CHECK(cuCtxSetCurrent(context));

    // Query device of current context.
    // This is the device of the provided stream.
    CUdevice device;
    OPEN3D_ML_CUDA_DRIVER_CHECK(cuCtxGetDevice(&device));

    // Restore previous context.
    OPEN3D_ML_CUDA_DRIVER_CHECK(cuCtxSetCurrent(current_context));

    // CUdevice is a typedef to int.
    return device;
}

class CUDAScopedDeviceStream {
public:
    explicit CUDAScopedDeviceStream(cudaStream_t stream)
        : scoped_device_(GetDevice(stream)), scoped_stream_(stream) {}

    CUDAScopedDeviceStream(CUDAScopedDeviceStream const &) = delete;
    void operator=(CUDAScopedDeviceStream const &) = delete;

private:
    core::CUDAScopedDevice scoped_device_;
    core::CUDAScopedStream scoped_stream_;
};
#endif

}  // namespace ml
}  // namespace open3d
