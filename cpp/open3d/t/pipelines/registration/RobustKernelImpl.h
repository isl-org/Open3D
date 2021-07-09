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

#pragma once

#include "open3d/core/CUDAUtils.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"

/// To use `Robust Kernel` functions please refer the unit-tests for
/// `t::registration` or the implementation use cases at
/// `t::pipelines::kernel::ComputePosePointToPlaneCUDA` and
/// `t::pipelines::kernel::ComputePosePointToPlaneCPU`.
///
/// \param METHOD registration::RobustKernelMethod Loss type.
/// \param scalar_t type: float / double.
/// \param scaling_parameter Scaling parameter for loss fine-tuning.
/// \param shape_parameter Shape parameter for Generalized Loss method.
#define DISPATCH_ROBUST_KERNEL_FUNCTION(METHOD, scalar_t, scaling_parameter,  \
                                        shape_parameter, ...)                 \
    [&] {                                                                     \
        if (METHOD ==                                                         \
            open3d::t::pipelines::registration::RobustKernelMethod::L2Loss) { \
            auto GetWeightFromRobustKernel =                                  \
                    [=] OPEN3D_HOST_DEVICE(scalar_t residual) -> scalar_t {   \
                return 1.0;                                                   \
            };                                                                \
            return __VA_ARGS__();                                             \
        } else if (METHOD == open3d::t::pipelines::registration::             \
                                     RobustKernelMethod::L1Loss) {            \
            auto GetWeightFromRobustKernel =                                  \
                    [=] OPEN3D_HOST_DEVICE(scalar_t residual) -> scalar_t {   \
                return 1.0 / OPEN3D_ABS(residual);                            \
            };                                                                \
            return __VA_ARGS__();                                             \
        } else if (METHOD == open3d::t::pipelines::registration::             \
                                     RobustKernelMethod::HuberLoss) {         \
            auto GetWeightFromRobustKernel =                                  \
                    [=] OPEN3D_HOST_DEVICE(scalar_t residual) -> scalar_t {   \
                return scaling_parameter /                                    \
                       OPEN3D_MAX(OPEN3D_ABS(residual), scaling_parameter);   \
            };                                                                \
            return __VA_ARGS__();                                             \
        } else if (METHOD == open3d::t::pipelines::registration::             \
                                     RobustKernelMethod::CauchyLoss) {        \
            auto GetWeightFromRobustKernel =                                  \
                    [=] OPEN3D_HOST_DEVICE(scalar_t residual) -> scalar_t {   \
                return 1.0 /                                                  \
                       (1.0 + OPEN3D_SQUARE(residual / scaling_parameter));   \
            };                                                                \
            return __VA_ARGS__();                                             \
        } else if (METHOD == open3d::t::pipelines::registration::             \
                                     RobustKernelMethod::GMLoss) {            \
            auto GetWeightFromRobustKernel =                                  \
                    [=] OPEN3D_HOST_DEVICE(scalar_t residual) -> scalar_t {   \
                return scaling_parameter /                                    \
                       OPEN3D_SQUARE(scaling_parameter +                      \
                                     OPEN3D_SQUARE(residual));                \
            };                                                                \
            return __VA_ARGS__();                                             \
        } else if (METHOD == open3d::t::pipelines::registration::             \
                                     RobustKernelMethod::TukeyLoss) {         \
            auto GetWeightFromRobustKernel =                                  \
                    [=] OPEN3D_HOST_DEVICE(scalar_t residual) -> scalar_t {   \
                return OPEN3D_SQUARE(                                         \
                        1.0 - OPEN3D_SQUARE(OPEN3D_MIN(                       \
                                      1.0, OPEN3D_ABS(residual) /             \
                                                   scaling_parameter)));      \
            };                                                                \
            return __VA_ARGS__();                                             \
        } else if (METHOD == open3d::t::pipelines::registration::             \
                                     RobustKernelMethod::GeneralizedLoss) {   \
            if (OPEN3D_IS_CLOSE(shape_parameter, 2.0, 1e-3)) {                \
                auto const_val = 1.0 / OPEN3D_SQUARE(scaling_parameter);      \
                auto GetWeightFromRobustKernel =                              \
                        [=] OPEN3D_HOST_DEVICE(                               \
                                scalar_t residual) -> scalar_t {              \
                    return const_val;                                         \
                };                                                            \
                return __VA_ARGS__();                                         \
            } else if (OPEN3D_IS_CLOSE(shape_parameter, 0.0, 1e-3)) {         \
                auto GetWeightFromRobustKernel =                              \
                        [=] OPEN3D_HOST_DEVICE(                               \
                                scalar_t residual) -> scalar_t {              \
                    return 2.0 / (OPEN3D_SQUARE(residual) +                   \
                                  2 * OPEN3D_SQUARE(scaling_parameter));      \
                };                                                            \
                return __VA_ARGS__();                                         \
            } else if (shape_parameter < -1e7) {                              \
                auto GetWeightFromRobustKernel =                              \
                        [=] OPEN3D_HOST_DEVICE(                               \
                                scalar_t residual) -> scalar_t {              \
                    return exp(OPEN3D_SQUARE(residual / scaling_parameter) /  \
                               (-2.0)) /                                      \
                           OPEN3D_SQUARE(scaling_parameter);                  \
                };                                                            \
                return __VA_ARGS__();                                         \
            } else {                                                          \
                auto GetWeightFromRobustKernel =                              \
                        [=] OPEN3D_HOST_DEVICE(                               \
                                scalar_t residual) -> scalar_t {              \
                    return pow((OPEN3D_SQUARE(residual / scaling_parameter) / \
                                        OPEN3D_ABS(shape_parameter - 2.0) +   \
                                1),                                           \
                               ((shape_parameter / 2.0) - 1.0)) /             \
                           OPEN3D_SQUARE(scaling_parameter);                  \
                };                                                            \
                return __VA_ARGS__();                                         \
            }                                                                 \
        } else {                                                              \
            utility::LogError("Unsupported method.");                         \
        }                                                                     \
    }()
