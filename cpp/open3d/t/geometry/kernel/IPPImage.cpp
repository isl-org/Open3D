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

#include "open3d/t/geometry/kernel/IPPImage.h"

#include <iw++/iw_image_color.hpp>
#include <iw++/iw_image_filter.hpp>
#include <iw++/iw_image_op.hpp>
#include <iw++/iw_image_transform.hpp>

#include "open3d/core/Dtype.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {
namespace ipp {

void To(const core::Tensor &src_im,
        core::Tensor &dst_im,
        double scale,
        double offset) {
    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.

    auto src_dtype = src_im.GetDtype();
    auto dst_dtype = dst_im.GetDtype();
    // Create IPP wrappers for all Open3D tensors
    const ::ipp::IwiImage ipp_src_im(
            ::ipp::IwiSize(src_im.GetShape(1), src_im.GetShape(0)),
            ToIppDataType(src_dtype), src_im.GetShape(2) /* channels */,
            0 /* border buffer size */, const_cast<void *>(src_im.GetDataPtr()),
            src_im.GetStride(0) * src_dtype.ByteSize());
    ::ipp::IwiImage ipp_dst_im(
            ::ipp::IwiSize(dst_im.GetShape(1), dst_im.GetShape(0)),
            ToIppDataType(dst_dtype), dst_im.GetShape(2) /* channels */,
            0 /* border buffer size */, dst_im.GetDataPtr(),
            dst_im.GetStride(0) * dst_dtype.ByteSize());

    try {
        ::ipp::iwiScale(ipp_src_im, ipp_dst_im, scale, offset);
    } catch (const ::ipp::IwException &e) {
        // See comments in icv/include/ippicv_types.h for m_status meaning
        utility::LogError("IPP-IW error {}: {}", e.m_status, e.m_string);
    }
}

void RGBToGray(const core::Tensor &src_im, core::Tensor &dst_im) {
    auto dtype = src_im.GetDtype();
    // Create IPP wrappers for all Open3D tensors
    const ::ipp::IwiImage ipp_src_im(
            ::ipp::IwiSize(src_im.GetShape(1), src_im.GetShape(0)),
            ToIppDataType(dtype), src_im.GetShape(2) /* channels */,
            0 /* border buffer size */, const_cast<void *>(src_im.GetDataPtr()),
            src_im.GetStride(0) * dtype.ByteSize());
    ::ipp::IwiImage ipp_dst_im(
            ::ipp::IwiSize(dst_im.GetShape(1), dst_im.GetShape(0)),
            ToIppDataType(dtype), dst_im.GetShape(2) /* channels */,
            0 /* border buffer size */, dst_im.GetDataPtr(),
            dst_im.GetStride(0) * dtype.ByteSize());

    try {
        ::ipp::iwiColorConvert(ipp_src_im, ::ipp::iwiColorRGB, ipp_dst_im,
                               ::ipp::iwiColorGray);
    } catch (const ::ipp::IwException &e) {
        // See comments in icv/include/ippicv_types.h for m_status meaning
        utility::LogError("IPP-IW error {}: {}", e.m_status, e.m_string);
    }
}

void Resize(const open3d::core::Tensor &src_im,
            open3d::core::Tensor &dst_im,
            int interp_type) {
    auto dtype = src_im.GetDtype();
    // Create IPP wrappers for all Open3D tensors
    const ::ipp::IwiImage ipp_src_im(
            ::ipp::IwiSize(src_im.GetShape(1), src_im.GetShape(0)),
            ToIppDataType(dtype), src_im.GetShape(2) /* channels */,
            0 /* border buffer size */, const_cast<void *>(src_im.GetDataPtr()),
            src_im.GetStride(0) * dtype.ByteSize());
    ::ipp::IwiImage ipp_dst_im(
            ::ipp::IwiSize(dst_im.GetShape(1), dst_im.GetShape(0)),
            ToIppDataType(dtype), dst_im.GetShape(2) /* channels */,
            0 /* border buffer size */, dst_im.GetDataPtr(),
            dst_im.GetStride(0) * dtype.ByteSize());

    static const std::unordered_map<int, IppiInterpolationType> type_dict = {
            {Image::Nearest, ippNearest}, {Image::Linear, ippLinear},
            {Image::Cubic, ippCubic},     {Image::Lanczos, ippLanczos},
            {Image::Super, ippSuper},
    };
    auto it = type_dict.find(interp_type);
    if (it == type_dict.end()) {
        utility::LogError("Unsupported interp type {}", interp_type);
    }

    try {
        utility::LogInfo("{}", it->second);
        ::ipp::iwiResize(ipp_src_im, ipp_dst_im, it->second);
    } catch (const ::ipp::IwException &e) {
        // See comments in icv/include/ippicv_types.h for m_status meaning
        utility::LogError("IPP-IW error {}: {}", e.m_status, e.m_string);
    }
}

void Dilate(const core::Tensor &src_im, core::Tensor &dst_im, int kernel_size) {
    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.

    // Create mask.
    core::Tensor mask =
            core::Tensor::Ones(core::SizeVector{kernel_size, kernel_size, 1},
                               core::Dtype::UInt8, src_im.GetDevice());

    auto dtype = src_im.GetDtype();
    // Create IPP wrappers for all Open3D tensors
    const ::ipp::IwiImage ipp_src_im(
            ::ipp::IwiSize(src_im.GetShape(1), src_im.GetShape(0)),
            ToIppDataType(dtype), src_im.GetShape(2) /* channels */,
            0 /* border buffer size */, const_cast<void *>(src_im.GetDataPtr()),
            src_im.GetStride(0) * dtype.ByteSize());
    ::ipp::IwiImage ipp_dst_im(
            ::ipp::IwiSize(dst_im.GetShape(1), dst_im.GetShape(0)),
            ToIppDataType(dtype), dst_im.GetShape(2) /* channels */,
            0 /* border buffer size */, dst_im.GetDataPtr(),
            dst_im.GetStride(0) * dtype.ByteSize());
    ::ipp::IwiImage ipp_mask_im(
            ::ipp::IwiSize(mask.GetShape(1), mask.GetShape(0)),
            ToIppDataType(mask.GetDtype()), mask.GetShape(2) /* channels */,
            0 /* border buffer size */, mask.GetDataPtr(),
            mask.GetStride(0) * mask.GetDtype().ByteSize());
    try {
        ::ipp::iwiFilterMorphology(
                ipp_src_im, ipp_dst_im, ::ipp::iwiMorphDilate, ipp_mask_im,
                ::ipp::IwDefault(), /* Do not use IwiFilterMorphologyParams() */
                ippBorderRepl);
    } catch (const ::ipp::IwException &e) {
        // See comments in icv/include/ippicv_types.h for m_status meaning
        utility::LogError("IPP-IW error {}: {}", e.m_status, e.m_string);
    }
}

void FilterBilateral(const core::Tensor &src_im,
                     core::Tensor &dst_im,
                     int kernel_size,
                     float value_sigma,
                     float dist_sigma) {
    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.
    auto dtype = src_im.GetDtype();

    // Create IPP wrappers for all Open3D tensors
    const ::ipp::IwiImage ipp_src_im(
            ::ipp::IwiSize(src_im.GetShape(1), src_im.GetShape(0)),
            ToIppDataType(dtype), src_im.GetShape(2) /* channels */,
            0 /* border buffer size */, const_cast<void *>(src_im.GetDataPtr()),
            src_im.GetStride(0) * dtype.ByteSize());
    ::ipp::IwiImage ipp_dst_im(
            ::ipp::IwiSize(dst_im.GetShape(1), dst_im.GetShape(0)),
            ToIppDataType(dtype), dst_im.GetShape(2) /* channels */,
            0 /* border buffer size */, dst_im.GetDataPtr(),
            dst_im.GetStride(0) * dtype.ByteSize());

    try {
        ::ipp::iwiFilterBilateral(ipp_src_im, ipp_dst_im, kernel_size / 2,
                                  value_sigma, dist_sigma);
    } catch (const ::ipp::IwException &e) {
        // See comments in icv/include/ippicv_types.h for m_status meaning
        utility::LogError("IPP-IW error {}: {}", e.m_status, e.m_string);
    }
}

void FilterGaussian(const core::Tensor &src_im,
                    core::Tensor &dst_im,
                    int kernel_size) {
    // Use a precomputed sigma to be consistent with npp:
    // https://docs.nvidia.com/cuda/npp/group__image__filter__gauss__border.html

    double sigma = 0.4 * (kernel_size / 2) * 0.6;
    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.
    auto dtype = src_im.GetDtype();

    // Create IPP wrappers for all Open3D tensors
    const ::ipp::IwiImage ipp_src_im(
            ::ipp::IwiSize(src_im.GetShape(1), src_im.GetShape(0)),
            ToIppDataType(dtype), src_im.GetShape(2) /* channels */,
            0 /* border buffer size */, const_cast<void *>(src_im.GetDataPtr()),
            src_im.GetStride(0) * dtype.ByteSize());
    ::ipp::IwiImage ipp_dst_im(
            ::ipp::IwiSize(dst_im.GetShape(1), dst_im.GetShape(0)),
            ToIppDataType(dtype), dst_im.GetShape(2) /* channels */,
            0 /* border buffer size */, dst_im.GetDataPtr(),
            dst_im.GetStride(0) * dtype.ByteSize());

    try {
        ::ipp::iwiFilterGaussian(ipp_src_im, ipp_dst_im, kernel_size, sigma);
    } catch (const ::ipp::IwException &e) {
        // See comments in icv/include/ippicv_types.h for m_status meaning
        utility::LogError("IPP-IW error {}: {}", e.m_status, e.m_string);
    }
}

void FilterSobel(const core::Tensor &src_im,
                 core::Tensor &dst_im_dx,
                 core::Tensor &dst_im_dy,
                 int kernel_size) {
    const static std::unordered_map<int, IppiMaskSize> kKernelSizeMap = {
            {3, ::ipp::ippMskSize3x3},
            {5, ::ipp::ippMskSize5x5},
    };
    auto it = kKernelSizeMap.find(kernel_size);
    if (it == kKernelSizeMap.end()) {
        utility::LogError("Unsupported size {} for IPP FilterSobel",
                          kernel_size);
    }

    // Create IPP wrappers for all Open3D tensors
    const ::ipp::IwiImage ipp_src_im(
            ::ipp::IwiSize(src_im.GetShape(1), src_im.GetShape(0)),
            ToIppDataType(src_im.GetDtype()), src_im.GetShape(2) /* channels */,
            0 /* border buffer size */, const_cast<void *>(src_im.GetDataPtr()),
            src_im.GetStride(0) * src_im.GetDtype().ByteSize());
    ::ipp::IwiImage ipp_dst_im_dx(
            ::ipp::IwiSize(dst_im_dx.GetShape(1), dst_im_dx.GetShape(0)),
            ToIppDataType(dst_im_dx.GetDtype()),
            dst_im_dx.GetShape(2) /* channels */, 0 /* border buffer size */,
            dst_im_dx.GetDataPtr(),
            dst_im_dx.GetStride(0) * dst_im_dx.GetDtype().ByteSize());
    ::ipp::IwiImage ipp_dst_im_dy(
            ::ipp::IwiSize(dst_im_dy.GetShape(1), dst_im_dy.GetShape(0)),
            ToIppDataType(dst_im_dy.GetDtype()),
            dst_im_dy.GetShape(2) /* channels */, 0 /* border buffer size */,
            dst_im_dy.GetDataPtr(),
            dst_im_dy.GetStride(0) * dst_im_dy.GetDtype().ByteSize());

    try {
        ::ipp::iwiFilterSobel(ipp_src_im, ipp_dst_im_dx,
                              IwiDerivativeType::iwiDerivVerFirst, it->second);
        ::ipp::iwiFilterSobel(ipp_src_im, ipp_dst_im_dy,
                              IwiDerivativeType::iwiDerivHorFirst, it->second);
    } catch (const ::ipp::IwException &e) {
        // See comments in icv/include/ippicv_types.h for m_status meaning
        utility::LogError("IPP-IW error {}: {}", e.m_status, e.m_string);
    }
}
}  // namespace ipp
}  // namespace geometry
}  // namespace t
}  // namespace open3d
