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

#include <iw++/iw_image_filter.hpp>

#include "open3d/core/Dtype.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {
namespace ipp {

void Dilate(const core::Tensor &src_im,
            core::Tensor &dstim,
            int half_kernel_size) {
    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.

    // Create mask.
    core::Tensor mask =
            core::Tensor::Ones(core::SizeVector{2 * half_kernel_size + 1,
                                                2 * half_kernel_size + 1, 1},
                               core::Dtype::UInt8, src_im.GetDevice());

    auto dtype = src_im.GetDtype();
    // Create IPP wrappers for all Open3D tensors
    const ::ipp::IwiImage ipp_src_im(
            ::ipp::IwiSize(src_im.GetShape(1), src_im.GetShape(0)),
            ToIppDataType(dtype), src_im.GetShape(2) /* channels */,
            0 /* border buffer size */, const_cast<void *>(src_im.GetDataPtr()),
            src_im.GetStride(0) * dtype.ByteSize());
    ::ipp::IwiImage ipp_dst_im(
            ::ipp::IwiSize(dstim.GetShape(1), dstim.GetShape(0)),
            ToIppDataType(dtype), dstim.GetShape(2) /* channels */,
            0 /* border buffer size */, dstim.GetDataPtr(),
            dstim.GetStride(0) * dtype.ByteSize());
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

}  // namespace ipp
}  // namespace geometry
}  // namespace t
}  // namespace open3d
