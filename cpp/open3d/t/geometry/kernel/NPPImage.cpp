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

void RGBToGray(const core::Tensor &src_im, core::Tensor &dst_im) {
    // create struct with ROI size
    NppiSize size_ROI = {static_cast<int>(dst_im.GetShape(1)),
                         static_cast<int>(dst_im.GetShape(0))};

    auto dtype = src_im.GetDtype();
#define NPP_ARGS                                           \
    static_cast<const npp_dtype *>(src_im.GetDataPtr()),   \
            src_im.GetStride(0) * dtype.ByteSize(),        \
            static_cast<npp_dtype *>(dst_im.GetDataPtr()), \
            dst_im.GetStride(0) * dtype.ByteSize(), size_ROI
    if (dtype == core::Dtype::UInt8) {
        using npp_dtype = Npp8u;
        nppiRGBToGray_8u_C3C1R(NPP_ARGS);
    } else if (dtype == core::Dtype::UInt16) {
        using npp_dtype = Npp16u;
        nppiRGBToGray_16u_C3C1R(NPP_ARGS);
    } else if (dtype == core::Dtype::Float32) {
        using npp_dtype = Npp32f;
        nppiRGBToGray_32f_C3C1R(NPP_ARGS);
    } else {
        utility::LogError("npp::FilterGaussian(): Unspported dtype {}",
                          dtype.ToString());
    }
#undef NPP_ARGS
}

void Resize(const open3d::core::Tensor &src_im,
            open3d::core::Tensor &dst_im,
            int interp_type) {
    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.
    NppiSize src_size = {static_cast<int>(src_im.GetShape(1)),
                         static_cast<int>(src_im.GetShape(0))};
    NppiRect src_roi = {0, 0, static_cast<int>(src_im.GetShape(1)),
                        static_cast<int>(src_im.GetShape(0))};

    // create struct with ROI size
    NppiSize dst_size = {static_cast<int>(dst_im.GetShape(1)),
                         static_cast<int>(dst_im.GetShape(0))};
    NppiRect dst_roi = {0, 0, static_cast<int>(dst_im.GetShape(1)),
                        static_cast<int>(dst_im.GetShape(0))};

    static const std::unordered_map<int, int> type_dict = {
            {Image::Nearest, NPPI_INTER_NN},
            {Image::Linear, NPPI_INTER_LINEAR},
            {Image::Cubic, NPPI_INTER_CUBIC},
            {Image::Lanczos, NPPI_INTER_LANCZOS},
            {Image::Super, NPPI_INTER_SUPER},
    };
    auto it = type_dict.find(interp_type);

    auto dtype = src_im.GetDtype();
#define NPP_ARGS                                                       \
    static_cast<const npp_dtype *>(src_im.GetDataPtr()),               \
            src_im.GetStride(0) * dtype.ByteSize(), src_size, src_roi, \
            static_cast<npp_dtype *>(dst_im.GetDataPtr()),             \
            dst_im.GetStride(0) * dtype.ByteSize(), dst_size, dst_roi, \
            it->second

    if (dtype == core::Dtype::UInt8) {
        using npp_dtype = Npp8u;
        if (src_im.GetShape(2) == 1) {
            nppiResize_8u_C1R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiResize_8u_C3R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiResize_8u_C4R(NPP_ARGS);
        }
    } else if (dtype == core::Dtype::UInt16) {
        using npp_dtype = Npp16u;
        if (src_im.GetShape(2) == 1) {
            nppiResize_16u_C1R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiResize_16u_C3R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiResize_16u_C4R(NPP_ARGS);
        }
    } else if (dtype == core::Dtype::Float32) {
        using npp_dtype = Npp32f;
        if (src_im.GetShape(2) == 1) {
            nppiResize_32f_C1R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiResize_32f_C3R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiResize_32f_C4R(NPP_ARGS);
        }
    } else {
        utility::LogError("npp::Resize(): Unspported dtype {}",
                          dtype.ToString());
    }
#undef NPP_ARGS
}

void Dilate(const core::Tensor &src_im, core::Tensor &dst_im, int kernel_size) {
    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.

    // Create mask.
    core::Tensor mask =
            core::Tensor::Ones(core::SizeVector{kernel_size, kernel_size, 1},
                               core::Dtype::UInt8, src_im.GetDevice());
    NppiSize mask_size = {kernel_size, kernel_size};

    NppiSize src_size = {static_cast<int>(src_im.GetShape(1)),
                         static_cast<int>(src_im.GetShape(0))};
    NppiPoint src_offset = {0, 0};

    // create struct with ROI size
    NppiSize size_ROI = {static_cast<int>(dst_im.GetShape(1)),
                         static_cast<int>(dst_im.GetShape(0))};
    NppiPoint anchor = {kernel_size / 2, kernel_size / 2};

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

void FilterBilateral(const core::Tensor &src_im,
                     core::Tensor &dst_im,
                     int kernel_size,
                     float value_sigma,
                     float dist_sigma) {
    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.
    NppiSize src_size = {static_cast<int>(src_im.GetShape(1)),
                         static_cast<int>(src_im.GetShape(0))};
    NppiPoint src_offset = {0, 0};

    // create struct with ROI size
    NppiSize size_ROI = {static_cast<int>(dst_im.GetShape(1)),
                         static_cast<int>(dst_im.GetShape(0))};

    auto dtype = src_im.GetDtype();
#define NPP_ARGS                                                               \
    static_cast<const npp_dtype *>(src_im.GetDataPtr()),                       \
            src_im.GetStride(0) * dtype.ByteSize(), src_size, src_offset,      \
            static_cast<npp_dtype *>(dst_im.GetDataPtr()),                     \
            dst_im.GetStride(0) * dtype.ByteSize(), size_ROI, kernel_size / 2, \
            1, value_sigma, dist_sigma, NPP_BORDER_REPLICATE
    if (dtype == core::Dtype::UInt8) {
        using npp_dtype = Npp8u;
        if (src_im.GetShape(2) == 1) {
            nppiFilterBilateralGaussBorder_8u_C1R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiFilterBilateralGaussBorder_8u_C3R(NPP_ARGS);
        }
    } else if (dtype == core::Dtype::UInt16) {
        using npp_dtype = Npp16u;
        if (src_im.GetShape(2) == 1) {
            nppiFilterBilateralGaussBorder_16u_C1R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiFilterBilateralGaussBorder_16u_C3R(NPP_ARGS);
        }
    } else if (dtype == core::Dtype::Float32) {
        using npp_dtype = Npp32f;
        if (src_im.GetShape(2) == 1) {
            nppiFilterBilateralGaussBorder_32f_C1R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiFilterBilateralGaussBorder_32f_C3R(NPP_ARGS);
        }
    } else {
        utility::LogError("npp::FilterBilateral(): Unspported dtype {}",
                          dtype.ToString());
    }
#undef NPP_ARGS
}

void FilterGaussian(const core::Tensor &src_im,
                    core::Tensor &dst_im,
                    int kernel_size) {
    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.
    NppiSize src_size = {static_cast<int>(src_im.GetShape(1)),
                         static_cast<int>(src_im.GetShape(0))};
    NppiPoint src_offset = {0, 0};

    // create struct with ROI size
    NppiSize size_ROI = {static_cast<int>(dst_im.GetShape(1)),
                         static_cast<int>(dst_im.GetShape(0))};

    auto dtype = src_im.GetDtype();
    const static std::unordered_map<int, NppiMaskSize> kKernelSizeMap = {
            {3, NPP_MASK_SIZE_3_X_3},    {5, NPP_MASK_SIZE_5_X_5},
            {7, NPP_MASK_SIZE_7_X_7},    {9, NPP_MASK_SIZE_9_X_9},
            {11, NPP_MASK_SIZE_11_X_11}, {13, NPP_MASK_SIZE_13_X_13},
            {15, NPP_MASK_SIZE_15_X_15},
    };
    auto it = kKernelSizeMap.find(kernel_size);
    if (it == kKernelSizeMap.end()) {
        utility::LogError("Unsupported size {} for NPP FilterGaussian",
                          kernel_size);
    }
#define NPP_ARGS                                                          \
    static_cast<const npp_dtype *>(src_im.GetDataPtr()),                  \
            src_im.GetStride(0) * dtype.ByteSize(), src_size, src_offset, \
            static_cast<npp_dtype *>(dst_im.GetDataPtr()),                \
            dst_im.GetStride(0) * dtype.ByteSize(), size_ROI, it->second, \
            NPP_BORDER_REPLICATE
    if (dtype == core::Dtype::UInt8) {
        using npp_dtype = Npp8u;
        if (src_im.GetShape(2) == 1) {
            nppiFilterGaussBorder_8u_C1R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiFilterGaussBorder_8u_C3R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiFilterGaussBorder_8u_C4R(NPP_ARGS);
        }
    } else if (dtype == core::Dtype::UInt16) {
        using npp_dtype = Npp16u;
        if (src_im.GetShape(2) == 1) {
            nppiFilterGaussBorder_16u_C1R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiFilterGaussBorder_16u_C3R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiFilterGaussBorder_16u_C4R(NPP_ARGS);
        }
    } else if (dtype == core::Dtype::Float32) {
        using npp_dtype = Npp32f;
        if (src_im.GetShape(2) == 1) {
            nppiFilterGaussBorder_32f_C1R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiFilterGaussBorder_32f_C3R(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiFilterGaussBorder_32f_C4R(NPP_ARGS);
        }
    } else {
        utility::LogError("npp::FilterGaussian(): Unspported dtype {}",
                          dtype.ToString());
    }
#undef NPP_ARGS
}

void FilterSobel(const core::Tensor &src_im,
                 core::Tensor &dst_im_dx,
                 core::Tensor &dst_im_dy,
                 int kernel_size) {
    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.
    NppiSize src_size = {static_cast<int>(src_im.GetShape(1)),
                         static_cast<int>(src_im.GetShape(0))};
    NppiPoint src_offset = {0, 0};

    // create struct with ROI size
    NppiSize size_ROI = {static_cast<int>(dst_im_dx.GetShape(1)),
                         static_cast<int>(dst_im_dx.GetShape(0))};
    auto dtype = src_im.GetDtype();
    const static std::unordered_map<int, NppiMaskSize> kKernelSizeMap = {
            {3, NPP_MASK_SIZE_3_X_3},
            {5, NPP_MASK_SIZE_5_X_5},
    };
    auto it = kKernelSizeMap.find(kernel_size);
    if (it == kKernelSizeMap.end()) {
        utility::LogError("Unsupported size {} for NPP FilterSobel",
                          kernel_size);
    }

    // Counterintuitive conventions: dy: Horizontal,  dx: Vertical.
    // Probable reason: dy detects horizontal edges, dx detects vertical edges.
#define NPP_ARGS_DX                                                       \
    static_cast<const npp_src_dtype *>(src_im.GetDataPtr()),              \
            src_im.GetStride(0) * dtype.ByteSize(), src_size, src_offset, \
            static_cast<npp_dst_dtype *>(dst_im_dx.GetDataPtr()),         \
            dst_im_dx.GetStride(0) * dst_im_dx.GetDtype().ByteSize(),     \
            size_ROI, it->second, NPP_BORDER_REPLICATE
#define NPP_ARGS_DY                                                       \
    static_cast<const npp_src_dtype *>(src_im.GetDataPtr()),              \
            src_im.GetStride(0) * dtype.ByteSize(), src_size, src_offset, \
            static_cast<npp_dst_dtype *>(dst_im_dy.GetDataPtr()),         \
            dst_im_dy.GetStride(0) * dst_im_dy.GetDtype().ByteSize(),     \
            size_ROI, it->second, NPP_BORDER_REPLICATE
    if (dtype == core::Dtype::UInt8) {
        using npp_src_dtype = Npp8u;
        using npp_dst_dtype = Npp16s;
        nppiFilterSobelVertBorder_8u16s_C1R(NPP_ARGS_DX);
        nppiFilterSobelHorizBorder_8u16s_C1R(NPP_ARGS_DY);
    } else if (dtype == core::Dtype::Float32) {
        using npp_src_dtype = Npp32f;
        using npp_dst_dtype = Npp32f;
        nppiFilterSobelVertMaskBorder_32f_C1R(NPP_ARGS_DX);
        nppiFilterSobelHorizMaskBorder_32f_C1R(NPP_ARGS_DY);
    } else {
        utility::LogError("npp::FilterSobel(): Unspported dtype {}",
                          dtype.ToString());
    }
#undef NPP_ARGS_DX
#undef NPP_ARGS_DY
}

}  // namespace npp
}  // namespace geometry
}  // namespace t
}  // namespace open3d
