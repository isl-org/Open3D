// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/kernel/IPPImage.h"

#include <iw++/iw_image_color.hpp>
#include <iw++/iw_image_filter.hpp>
#include <iw++/iw_image_op.hpp>
#include <iw++/iw_image_transform.hpp>

#include "open3d/core/Dtype.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/utility/Logging.h"

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
            t::geometry::Image::InterpType interp_type) {
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

    static const std::unordered_map<t::geometry::Image::InterpType,
                                    IppiInterpolationType>
            type_dict = {
                    {t::geometry::Image::InterpType::Nearest, ippNearest},
                    {t::geometry::Image::InterpType::Linear, ippLinear},
                    {t::geometry::Image::InterpType::Cubic, ippCubic},
                    {t::geometry::Image::InterpType::Lanczos, ippLanczos},
                    {t::geometry::Image::InterpType::Super, ippSuper},
            };

    auto it = type_dict.find(interp_type);
    if (it == type_dict.end()) {
        utility::LogError("Unsupported interp type {}",
                          static_cast<int>(interp_type));
    }

    try {
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
                               core::UInt8, src_im.GetDevice());

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

void Filter(const open3d::core::Tensor &src_im,
            open3d::core::Tensor &dst_im,
            const open3d::core::Tensor &kernel) {
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
    ::ipp::IwiImage ipp_kernel(
            ::ipp::IwiSize(kernel.GetShape(1), kernel.GetShape(0)),
            ToIppDataType(core::Float32), 1 /* channels */,
            0 /* border buffer size */, const_cast<void *>(kernel.GetDataPtr()),
            kernel.GetStride(0) * core::Float32.ByteSize());

    try {
        ::ipp::iwiFilter(ipp_src_im, ipp_dst_im, ipp_kernel);
    } catch (const ::ipp::IwException &e) {
        // See comments in icv/include/ippicv_types.h for m_status meaning
        utility::LogError("IPP-IW error {}: {}", e.m_status, e.m_string);
    }
};

void FilterBilateral(const core::Tensor &src_im,
                     core::Tensor &dst_im,
                     int kernel_size,
                     float value_sigma,
                     float distance_sigma) {
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
                                  value_sigma * value_sigma,
                                  distance_sigma * distance_sigma);
    } catch (const ::ipp::IwException &e) {
        // See comments in icv/include/ippicv_types.h for m_status meaning
        utility::LogError("IPP-IW error {}: {}", e.m_status, e.m_string);
    }
}

void FilterGaussian(const core::Tensor &src_im,
                    core::Tensor &dst_im,
                    int kernel_size,
                    float sigma) {
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
        utility::LogError("Unsupported kernel size {} for IPP FilterSobel",
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
        // IPP uses a "left minus right" kernel,
        // https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing/filtering-functions-2/fixed-filters/filtersobel.html
        // so we need to negate it in-place.
        dst_im_dx.Neg_();
    } catch (const ::ipp::IwException &e) {
        // See comments in icv/include/ippicv_types.h for m_status meaning
        utility::LogError("IPP-IW error {}: {}", e.m_status, e.m_string);
    }
}
}  // namespace ipp
}  // namespace geometry
}  // namespace t
}  // namespace open3d
