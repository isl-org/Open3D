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

#include <cmath>

#include "open3d/core/CUDAUtils.h"
#include "open3d/t/geometry/kernel/GeometryMacros.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"

#ifndef __CUDACC__
using std::abs;
using std::exp;
using std::max;
using std::min;
using std::pow;
#endif

using open3d::t::pipelines::registration::RobustKernelMethod;

/// To use `Robust Kernel` functions please refer the unit-tests for
/// `t::registration` or the implementation use cases at
/// `t::pipelines::kernel::ComputePosePointToPlaneCUDA` and
/// `t::pipelines::kernel::ComputePosePointToPlaneCPU`.
///
/// \param METHOD registration::RobustKernelMethod Loss type.
/// \param scalar_t type: float / double.
/// \param scaling_parameter Scaling parameter for loss fine-tuning.
/// \param shape_parameter Shape parameter for Generalized Loss method.
#define DISPATCH_ROBUST_KERNEL_FUNCTION(METHOD, scalar_t, scaling_parameter, \
                                        shape_parameter, ...)                \
    [&] {                                                                    \
        scalar_t scale = static_cast<scalar_t>(scaling_parameter);           \
        if (METHOD == RobustKernelMethod::L2Loss) {                          \
            auto GetWeightFromRobustKernel =                                 \
                    [=] OPEN3D_HOST_DEVICE(scalar_t residual) -> scalar_t {  \
                return 1.0;                                                  \
            };                                                               \
            return __VA_ARGS__();                                            \
        } else if (METHOD == RobustKernelMethod::L1Loss) {                   \
            auto GetWeightFromRobustKernel =                                 \
                    [=] OPEN3D_HOST_DEVICE(scalar_t residual) -> scalar_t {  \
                return 1.0 / abs(residual);                                  \
            };                                                               \
            return __VA_ARGS__();                                            \
        } else if (METHOD == RobustKernelMethod::HuberLoss) {                \
            auto GetWeightFromRobustKernel =                                 \
                    [=] OPEN3D_HOST_DEVICE(scalar_t residual) -> scalar_t {  \
                return scale / max(abs(residual), scale);                    \
            };                                                               \
            return __VA_ARGS__();                                            \
        } else if (METHOD == RobustKernelMethod::CauchyLoss) {               \
            auto GetWeightFromRobustKernel =                                 \
                    [=] OPEN3D_HOST_DEVICE(scalar_t residual) -> scalar_t {  \
                return 1.0 / (1.0 + Square(residual / scale));               \
            };                                                               \
            return __VA_ARGS__();                                            \
        } else if (METHOD == RobustKernelMethod::GMLoss) {                   \
            auto GetWeightFromRobustKernel =                                 \
                    [=] OPEN3D_HOST_DEVICE(scalar_t residual) -> scalar_t {  \
                return scale / Square(scale + Square(residual));             \
            };                                                               \
            return __VA_ARGS__();                                            \
        } else if (METHOD == RobustKernelMethod::TukeyLoss) {                \
            auto GetWeightFromRobustKernel =                                 \
                    [=] OPEN3D_HOST_DEVICE(scalar_t residual) -> scalar_t {  \
                return Square(1.0 - Square(min((scalar_t)1.0,                \
                                               abs(residual) / scale)));     \
            };                                                               \
            return __VA_ARGS__();                                            \
        } else if (METHOD == RobustKernelMethod::GeneralizedLoss) {          \
            if (open3d::IsClose(shape_parameter, 2.0, 1e-3)) {               \
                auto const_val = 1.0 / Square(scale);                        \
                auto GetWeightFromRobustKernel =                             \
                        [=] OPEN3D_HOST_DEVICE(                              \
                                scalar_t residual) -> scalar_t {             \
                    return const_val;                                        \
                };                                                           \
                return __VA_ARGS__();                                        \
            } else if (open3d::IsClose(shape_parameter, 0.0, 1e-3)) {        \
                auto GetWeightFromRobustKernel =                             \
                        [=] OPEN3D_HOST_DEVICE(                              \
                                scalar_t residual) -> scalar_t {             \
                    return 2.0 / (Square(residual) + 2 * Square(scale));     \
                };                                                           \
                return __VA_ARGS__();                                        \
            } else if (shape_parameter < -1e7) {                             \
                auto GetWeightFromRobustKernel =                             \
                        [=] OPEN3D_HOST_DEVICE(                              \
                                scalar_t residual) -> scalar_t {             \
                    return exp(Square(residual / scale) / (-2.0)) /          \
                           Square(scale);                                    \
                };                                                           \
                return __VA_ARGS__();                                        \
            } else {                                                         \
                auto GetWeightFromRobustKernel =                             \
                        [=] OPEN3D_HOST_DEVICE(                              \
                                scalar_t residual) -> scalar_t {             \
                    return pow((Square(residual / scale) /                   \
                                        abs(shape_parameter - 2.0) +         \
                                1),                                          \
                               ((shape_parameter / 2.0) - 1.0)) /            \
                           Square(scale);                                    \
                };                                                           \
                return __VA_ARGS__();                                        \
            }                                                                \
        } else {                                                             \
            utility::LogError("Unsupported method.");                        \
        }                                                                    \
    }()
