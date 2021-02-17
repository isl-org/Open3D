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

#include "open3d/t/geometry/kernel/NPPImage.h"

#include <nppdefs.h>
#include <nppi.h>

#include "open3d/core/Dtype.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/utility/Console.h"

namespace open3d {
namespace t {
namespace geometry {
namespace npp {

void Dilate(const core::Tensor &src_im,
            core::Tensor &dst_im,
            int half_kernel_size) {
    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.

    // Create mask.
    core::Tensor mask =
            core::Tensor::Ones(core::SizeVector{2 * half_kernel_size + 1,
                                                2 * half_kernel_size + 1, 1},
                               core::Dtype::UInt8, src_im.GetDevice());
    NppiSize mask_size = {2 * half_kernel_size + 1, 2 * half_kernel_size + 1};

    NppiSize src_size = {static_cast<int>(src_im.GetShape(1)),
                         static_cast<int>(src_im.GetShape(0))};
    NppiPoint src_offset = {0, 0};

    // create struct with ROI size
    NppiSize size_ROI = {static_cast<int>(dst_im.GetShape(1)),
                         static_cast<int>(dst_im.GetShape(0))};
    NppiPoint anchor = {half_kernel_size, half_kernel_size};

    auto dtype = src_im.GetDtype();
#define NPP_ARGS                                                          \
    static_cast<const npp_dtype *>(src_im.GetDataPtr()),                  \
            src_im.GetStride(0) * dtype.ByteSize(), src_size, src_offset, \
            static_cast<npp_dtype *>(dst_im.GetDataPtr()),                \
            dst_im.GetStride(0) * dtype.ByteSize(), size_ROI,             \
            static_cast<const uint8_t *>(mask.GetDataPtr()), mask_size,   \
            anchor, NPP_BORDER_REPLICATE
    if (dtype == core::Dtype::Bool || dtype == core::Dtype::UInt8) {
        using npp_dtype = Npp8u;
        if (src_im.GetShape(2) == 1) {
            nppiDilateBorder_8u_C1R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiDilateBorder_8u_C3R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiDilateBorder_8u_C4R(NPP_ARGS);
        }
    } else if (dtype == core::Dtype::UInt16) {
        using npp_dtype = Npp16u;
        if (src_im.GetShape(2) == 1) {
            nppiDilateBorder_16u_C1R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiDilateBorder_16u_C3R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiDilateBorder_16u_C4R(NPP_ARGS);
        }
    } else if (dtype == core::Dtype::Float32) {
        using npp_dtype = Npp32f;
        if (src_im.GetShape(2) == 1) {
            nppiDilateBorder_32f_C1R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiDilateBorder_32f_C3R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiDilateBorder_32f_C4R(NPP_ARGS);
        }
    } else {
        utility::LogError("npp::Dilate(): Unspported dtype {}",
                          dtype.ToString());
    }
#undef NPP_ARGS
}

}  // namespace npp
}  // namespace geometry
}  // namespace t
}  // namespace open3d
