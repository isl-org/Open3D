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

#pragma once

#include "open3d/core/CUDAUtils.h"
#include "open3d/t/pipelines/registration/RobustKernel.h"

#define O3D_ABS(a) (a < 0 ? -a : a)
#define O3D_MAX(a, b) (a > b ? a : b)
#define O3D_MIN(a, b) (a < b ? a : b)
#define O3D_SQUARE(a) ((a) * (a))

#define DISPATCH_ROBUST_KERNEL_FUNCTION(METHOD, T, k, c, ...)                 \
    [&] {                                                                     \
        if (METHOD ==                                                         \
            open3d::t::pipelines::registration::RobustKernelMethod::L2Loss) { \
            auto func_t = [=] OPEN3D_HOST_DEVICE(T r) -> T { return 1.0; };   \
            return __VA_ARGS__();                                             \
        } else if (METHOD == open3d::t::pipelines::registration::             \
                                     RobustKernelMethod::L1Loss) {            \
            auto func_t = [=] OPEN3D_HOST_DEVICE(T r) -> T {                  \
                return 1.0 / O3D_ABS(r);                                      \
            };                                                                \
            return __VA_ARGS__();                                             \
        } else if (METHOD == open3d::t::pipelines::registration::             \
                                     RobustKernelMethod::HuberLoss) {         \
            auto func_t = [=] OPEN3D_HOST_DEVICE(T r) -> T {                  \
                return k / O3D_MAX(O3D_ABS(r), k);                            \
            };                                                                \
            return __VA_ARGS__();                                             \
        } else if (METHOD == open3d::t::pipelines::registration::             \
                                     RobustKernelMethod::CauchyLoss) {        \
            auto func_t = [=] OPEN3D_HOST_DEVICE(T r) -> T {                  \
                return 1.0 / (1.0 + O3D_SQUARE(r / k));                       \
            };                                                                \
            return __VA_ARGS__();                                             \
        } else if (METHOD == open3d::t::pipelines::registration::             \
                                     RobustKernelMethod::GMLoss) {            \
            auto func_t = [=] OPEN3D_HOST_DEVICE(T r) -> T {                  \
                return k / O3D_SQUARE(k + O3D_SQUARE(r));                     \
            };                                                                \
            return __VA_ARGS__();                                             \
        } else if (METHOD == open3d::t::pipelines::registration::             \
                                     RobustKernelMethod::TukeyLoss) {         \
            auto func_t = [=] OPEN3D_HOST_DEVICE(T r) -> T {                  \
                return O3D_SQUARE(1.0 -                                       \
                                  O3D_SQUARE(O3D_MIN(1.0, O3D_ABS(r) / k)));  \
            };                                                                \
            return __VA_ARGS__();                                             \
        } else if (METHOD == open3d::t::pipelines::registration::             \
                                     RobustKernelMethod::GeneralizedLoss) {   \
            auto func_t = [=] OPEN3D_HOST_DEVICE(T r) -> T {                  \
                return 1.0 / O3D_ABS(r);                                      \
            };                                                                \
            return __VA_ARGS__();                                             \
        } else {                                                              \
            utility::LogError("Unsupported method.");                         \
        }                                                                     \
    }()
