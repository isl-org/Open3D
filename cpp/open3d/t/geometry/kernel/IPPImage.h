// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#pragma once

#ifdef WITH_IPP
// Not available for Remap
// Auto-enable multi-threaded implementations
// #define IPP_ENABLED_THREADING_LAYER_REDEFINITIONS 1
#define IPP_CALL(ipp_function, ...) ipp_function(__VA_ARGS__);

#if IPP_VERSION_INT < \
        20211000  // macOS IPP v2021.9.11 uses old directory layout
#include <iw++/iw_core.hpp>
#else  // Linux and Windows IPP v2021.10+ uses new directory layout
#include <ipp/iw++/iw_core.hpp>
#endif

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"

namespace open3d {
namespace t {
namespace geometry {
namespace ipp {

inline ::ipp::IppDataType ToIppDataType(core::Dtype dtype) {
    if (dtype == core::UInt8 || dtype == core::Bool) {
        return ipp8u;
    } else if (dtype == core::UInt16) {
        return ipp16u;
    } else if (dtype == core::Int16) {
        return ipp16s;
    } else if (dtype == core::Int32) {
        return ipp32s;
    } else if (dtype == core::Int64) {
        return ipp64s;
    } else if (dtype == core::Float32) {
        return ipp32f;
    } else if (dtype == core::Float64) {
        return ipp64f;
    } else {
        return ippUndef;
    }
}

void To(const core::Tensor &src_im,
        core::Tensor &dst_im,
        double scale,
        double offset);

void RGBToGray(const core::Tensor &src_im, core::Tensor &dst_im);

void Dilate(const core::Tensor &srcim, core::Tensor &dstim, int kernel_size);

void Resize(const core::Tensor &srcim,
            core::Tensor &dstim,
            t::geometry::Image::InterpType interp_type);

void Filter(const core::Tensor &srcim,
            core::Tensor &dstim,
            const core::Tensor &kernel);

void FilterBilateral(const core::Tensor &srcim,
                     core::Tensor &dstim,
                     int kernel_size,
                     float value_sigma,
                     float distance_sigma);

void FilterGaussian(const core::Tensor &srcim,
                    core::Tensor &dstim,
                    int kernel_size,
                    float sigma);

void FilterSobel(const core::Tensor &srcim,
                 core::Tensor &dstim_dx,
                 core::Tensor &dstim_dy,
                 int kernel_size);

void Remap(const core::Tensor &src_im,       /*{Ws, Hs, C}*/
           const core::Tensor &dst2src_xmap, /*{Wd, Hd}, float*/
           const core::Tensor &dst2src_ymap, /*{Wd, Hd, 2}, float*/
           core::Tensor &dst_im,             /*{Wd, Hd, 2}*/
           Image::InterpType interp_type);

}  // namespace ipp
}  // namespace geometry
}  // namespace t
}  // namespace open3d

#else
#define IPP_CALL(ipp_function, ...) \
    utility::LogError("Not built with IPP-IW, cannot call " #ipp_function);
#endif  // WITH_IPP
