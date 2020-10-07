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

#include "open3d/core/Dtype.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/utility/Console.h"

#ifdef BUILD_CUDA_MODULE

#include <nppi.h>

namespace open3d {
namespace t {
namespace geometry {
namespace npp {

void dilate(const core::Tensor &srcim,
            core::Tensor &dstim,
            int half_kernel_size) {
    if (srcim.GetDevice().GetType() != core::Device::DeviceType::CUDA)
        utility::LogError("NPP functions need CUDA tensors.");
    if (!supported(srcim.GetDtype(), srcim.GetChannels()))
        utility::LogError(
                "NPP does not support image with data type {} with {} channels",
                srcim.GetDtype.ToString(), srcim.GetChannels());
    // create nask
    Tensor mask(2 * half_kernel_size + 1, 2 * half_kernel_size + 1, 1,
                srcim.GetDtype(), srcim.GetDevice());
    mask.Fill(1);
    NppiSize oMaskSize = {2 * half_kernel_size + 1, 2 * half_kernel_size + 1};

    NppiSize oSrcSize = {srcim.GetShape(1), srcim.GetShape(0)};
    NppiPoint oSrcOffset = {0, 0};

    // create struct with ROI size
    NppiSize oSizeROI = {dstim.GetShape(1), dstim.GetShape(0)};
    NppiPoint oAnchor = {half_kernel_size, half_kernel_size};

    switch (srcim.GetDtype()) {
        case core::DType::Uint8:

            switch (srcim.GetShape(2)) {
                case 1:
                    nppiDilateBorder_8u_C1R(
                            srcim.GetsrcimPtr(), (int32_t)srcim.GetStride(0),
                            oSrcSize, oSrcOffset, dstim.GetDataPtr(),
                            dstim.GetStride(0), oSizeROI, mask.GetDataPtr(),
                            oMaskSize, oAnchor, NPP_BORDER_REPLICATE);

                    break;
                case 3:
                    nppiDilateBorder_8u_C3R(
                            srcim.GetsrcimPtr(), (int32_t)srcim.GetStride(0),
                            oSrcSize, oSrcOffset, dstim.GetDataPtr(),
                            dstim.GetStride(0), oSizeROI, mask.GetDataPtr(),
                            oMaskSize, oAnchor, NPP_BORDER_REPLICATE);

                    break;
                case 4:
                    nppiDilateBorder_8u_C4R(
                            srcim.GetsrcimPtr(), (int32_t)srcim.GetStride(0),
                            oSrcSize, oSrcOffset, dstim.GetDataPtr(),
                            dstim.GetStride(0), oSizeROI, mask.GetDataPtr(),
                            oMaskSize, oAnchor, NPP_BORDER_REPLICATE);

                    break;
            }
            break;

        default:
    }
}

}  // namespace npp
}  // namespace geometry
}  // namespace t
}  // namespace open3d

#else  // BUILD_CUDA_MODULE

namespace open3d {
namespace t {
namespace geometry {
namespace npp {

void dilate(const core::Tensor &srcim,
            core::Tensor &dstim,
            int half_kernel_size) {
    utility::LogError("NPP not available since Open3D was built without CUDA");
}

}  // namespace npp
}  // namespace geometry
}  // namespace t
}  // namespace open3d

#endif  // BUILD_CUDA_MODULE
