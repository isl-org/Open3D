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

#ifdef WITH_IPPICV
#define IPP_CALL(ipp_function, ...) ipp_function(__VA_ARGS__);

// Required by IPPICV headers, defined here to keep other compile commands clean
#define ICV_BASE
#define IW_BUILD
#include <iw++/iw_core.hpp>

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
}  // namespace ipp
}  // namespace geometry
}  // namespace t
}  // namespace open3d

#else
#define IPP_CALL(ipp_function, ...) \
    utility::LogError("Not built with IPP-IW, cannot call " #ipp_function);
#endif  // WITH_IPPICV
