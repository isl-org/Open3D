// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

#ifdef BUILD_CUDA_MODULE
#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"

namespace open3d {
namespace t {
namespace geometry {
namespace npp {

void RGBToGray(const core::Tensor &src_im, core::Tensor &dst_im);

void Dilate(const open3d::core::Tensor &srcim,
            open3d::core::Tensor &dstim,
            int kernel_size);

void Resize(const open3d::core::Tensor &srcim,
            open3d::core::Tensor &dstim,
            t::geometry::Image::InterpType interp_type);

void Filter(const open3d::core::Tensor &srcim,
            open3d::core::Tensor &dstim,
            const open3d::core::Tensor &kernel);

void FilterBilateral(const open3d::core::Tensor &srcim,
                     open3d::core::Tensor &dstim,
                     int kernel_size,
                     float value_sigma,
                     float distance_sigma);

void FilterGaussian(const open3d::core::Tensor &srcim,
                    open3d::core::Tensor &dstim,
                    int kernel_size,
                    float sigma);

void FilterSobel(const open3d::core::Tensor &srcim,
                 open3d::core::Tensor &dstim_dx,
                 open3d::core::Tensor &dstim_dy,
                 int kernel_size);
}  // namespace npp
}  // namespace geometry
}  // namespace t
}  // namespace open3d

#endif  // BUILD_CUDA_MODULE
