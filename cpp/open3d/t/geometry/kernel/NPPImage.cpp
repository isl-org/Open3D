// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/kernel/NPPImage.h"

#include <npp.h>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace t {
namespace geometry {
namespace npp {

static NppStreamContext MakeNPPContext() {
    NppStreamContext context;
    context.hStream = core::cuda::GetStream();
    context.nCudaDeviceId = core::cuda::GetDevice();

    cudaDeviceProp device_prop;
    OPEN3D_CUDA_CHECK(
            cudaGetDeviceProperties(&device_prop, core::cuda::GetDevice()));

    context.nMultiProcessorCount = device_prop.multiProcessorCount;
    context.nMaxThreadsPerMultiProcessor =
            device_prop.maxThreadsPerMultiProcessor;
    context.nSharedMemPerBlock = device_prop.sharedMemPerBlock;

    int cc_major;
    OPEN3D_CUDA_CHECK(cudaDeviceGetAttribute(&cc_major,
                                             cudaDevAttrComputeCapabilityMajor,
                                             core::cuda::GetDevice()));
    context.nCudaDevAttrComputeCapabilityMajor = cc_major;

    int cc_minor;
    OPEN3D_CUDA_CHECK(cudaDeviceGetAttribute(&cc_minor,
                                             cudaDevAttrComputeCapabilityMinor,
                                             core::cuda::GetDevice()));
    context.nCudaDevAttrComputeCapabilityMinor = cc_minor;

// The NPP documentation incorrectly states that nStreamFlags becomes available
// in NPP 10.2 (CUDA 10.2). Instead, NPP 11.1 (CUDA 11.0) is the first release
// to expose this member variable.
#if NPP_VERSION >= 11100
    unsigned int stream_flags;
    OPEN3D_CUDA_CHECK(
            cudaStreamGetFlags(core::cuda::GetStream(), &stream_flags));
    context.nStreamFlags = stream_flags;
#endif

    return context;
}

void RGBToGray(const core::Tensor &src_im, core::Tensor &dst_im) {
    if (src_im.GetDevice() != dst_im.GetDevice()) {
        utility::LogError(
                "src_im and dst_im are not on the same device, got {} and {}.",
                src_im.GetDevice().ToString(), dst_im.GetDevice().ToString());
    }
    core::CUDAScopedDevice scoped_device(src_im.GetDevice());

    NppiSize size_ROI = {static_cast<int>(dst_im.GetShape(1)),
                         static_cast<int>(dst_im.GetShape(0))};

    auto dtype = src_im.GetDtype();
    auto context = MakeNPPContext();
#define NPP_ARGS                                           \
    static_cast<const npp_dtype *>(src_im.GetDataPtr()),   \
            src_im.GetStride(0) * dtype.ByteSize(),        \
            static_cast<npp_dtype *>(dst_im.GetDataPtr()), \
            dst_im.GetStride(0) * dtype.ByteSize(), size_ROI, context
    if (dtype == core::UInt8) {
        using npp_dtype = Npp8u;
        nppiRGBToGray_8u_C3C1R_Ctx(NPP_ARGS);
    } else if (dtype == core::UInt16) {
        using npp_dtype = Npp16u;
        nppiRGBToGray_16u_C3C1R_Ctx(NPP_ARGS);
    } else if (dtype == core::Float32) {
        using npp_dtype = Npp32f;
        nppiRGBToGray_32f_C3C1R_Ctx(NPP_ARGS);
    } else {
        utility::LogError("npp::FilterGaussian(): Unsupported dtype {}",
                          dtype.ToString());
    }
#undef NPP_ARGS
}

void Resize(const open3d::core::Tensor &src_im,
            open3d::core::Tensor &dst_im,
            t::geometry::Image::InterpType interp_type) {
    if (src_im.GetDevice() != dst_im.GetDevice()) {
        utility::LogError(
                "src_im and dst_im are not on the same device, got {} and {}.",
                src_im.GetDevice().ToString(), dst_im.GetDevice().ToString());
    }
    core::CUDAScopedDevice scoped_device(src_im.GetDevice());

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

    static const std::unordered_map<t::geometry::Image::InterpType, int>
            type_dict = {
                    {t::geometry::Image::InterpType::Nearest, NPPI_INTER_NN},
                    {t::geometry::Image::InterpType::Linear, NPPI_INTER_LINEAR},
                    {t::geometry::Image::InterpType::Cubic, NPPI_INTER_CUBIC},
                    {t::geometry::Image::InterpType::Lanczos,
                     NPPI_INTER_LANCZOS},
                    {t::geometry::Image::InterpType::Super, NPPI_INTER_SUPER},
            };
    auto it = type_dict.find(interp_type);
    if (it == type_dict.end()) {
        utility::LogError("Invalid interpolation type {}.",
                          static_cast<int>(interp_type));
    }

    auto dtype = src_im.GetDtype();
    auto context = MakeNPPContext();
#define NPP_ARGS                                                       \
    static_cast<const npp_dtype *>(src_im.GetDataPtr()),               \
            src_im.GetStride(0) * dtype.ByteSize(), src_size, src_roi, \
            static_cast<npp_dtype *>(dst_im.GetDataPtr()),             \
            dst_im.GetStride(0) * dtype.ByteSize(), dst_size, dst_roi, \
            it->second, context

    if (dtype == core::UInt8) {
        using npp_dtype = Npp8u;
        if (src_im.GetShape(2) == 1) {
            nppiResize_8u_C1R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiResize_8u_C3R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiResize_8u_C4R_Ctx(NPP_ARGS);
        }
    } else if (dtype == core::UInt16) {
        using npp_dtype = Npp16u;
        if (src_im.GetShape(2) == 1) {
            nppiResize_16u_C1R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiResize_16u_C3R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiResize_16u_C4R_Ctx(NPP_ARGS);
        }
    } else if (dtype == core::Float32) {
        using npp_dtype = Npp32f;
        if (src_im.GetShape(2) == 1) {
            nppiResize_32f_C1R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiResize_32f_C3R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiResize_32f_C4R_Ctx(NPP_ARGS);
        }
    } else {
        utility::LogError("npp::Resize(): Unsupported dtype {}",
                          dtype.ToString());
    }
#undef NPP_ARGS
}

void Dilate(const core::Tensor &src_im, core::Tensor &dst_im, int kernel_size) {
    if (src_im.GetDevice() != dst_im.GetDevice()) {
        utility::LogError(
                "src_im and dst_im are not on the same device, got {} and {}.",
                src_im.GetDevice().ToString(), dst_im.GetDevice().ToString());
    }
    core::CUDAScopedDevice scoped_device(src_im.GetDevice());
    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.

    // Create mask.
    core::Tensor mask =
            core::Tensor::Ones(core::SizeVector{kernel_size, kernel_size, 1},
                               core::UInt8, src_im.GetDevice());
    NppiSize mask_size = {kernel_size, kernel_size};

    NppiSize src_size = {static_cast<int>(src_im.GetShape(1)),
                         static_cast<int>(src_im.GetShape(0))};
    NppiPoint src_offset = {0, 0};

    // create struct with ROI size
    NppiSize size_ROI = {static_cast<int>(dst_im.GetShape(1)),
                         static_cast<int>(dst_im.GetShape(0))};
    NppiPoint anchor = {kernel_size / 2, kernel_size / 2};

    auto dtype = src_im.GetDtype();
    auto context = MakeNPPContext();
#define NPP_ARGS                                                          \
    static_cast<const npp_dtype *>(src_im.GetDataPtr()),                  \
            src_im.GetStride(0) * dtype.ByteSize(), src_size, src_offset, \
            static_cast<npp_dtype *>(dst_im.GetDataPtr()),                \
            dst_im.GetStride(0) * dtype.ByteSize(), size_ROI,             \
            static_cast<const uint8_t *>(mask.GetDataPtr()), mask_size,   \
            anchor, NPP_BORDER_REPLICATE, context
    if (dtype == core::Bool || dtype == core::UInt8) {
        using npp_dtype = Npp8u;
        if (src_im.GetShape(2) == 1) {
            nppiDilateBorder_8u_C1R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiDilateBorder_8u_C3R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiDilateBorder_8u_C4R_Ctx(NPP_ARGS);
        }
    } else if (dtype == core::UInt16) {
        using npp_dtype = Npp16u;
        if (src_im.GetShape(2) == 1) {
            nppiDilateBorder_16u_C1R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiDilateBorder_16u_C3R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiDilateBorder_16u_C4R_Ctx(NPP_ARGS);
        }
    } else if (dtype == core::Float32) {
        using npp_dtype = Npp32f;
        if (src_im.GetShape(2) == 1) {
            nppiDilateBorder_32f_C1R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiDilateBorder_32f_C3R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiDilateBorder_32f_C4R_Ctx(NPP_ARGS);
        }
    } else {
        utility::LogError("npp::Dilate(): Unsupported dtype {}",
                          dtype.ToString());
    }
#undef NPP_ARGS
}

void Filter(const open3d::core::Tensor &src_im,
            open3d::core::Tensor &dst_im,
            const open3d::core::Tensor &kernel) {
    if (src_im.GetDevice() != dst_im.GetDevice()) {
        utility::LogError(
                "src_im and dst_im are not on the same device, got {} and {}.",
                src_im.GetDevice().ToString(), dst_im.GetDevice().ToString());
    }
    core::CUDAScopedDevice scoped_device(src_im.GetDevice());

    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.
    NppiSize src_size = {static_cast<int>(src_im.GetShape(1)),
                         static_cast<int>(src_im.GetShape(0))};
    NppiPoint src_offset = {0, 0};

    // create struct with ROI size
    NppiSize size_ROI = {static_cast<int>(dst_im.GetShape(1)),
                         static_cast<int>(dst_im.GetShape(0))};

    // Generate separable kernel weights given the sigma value.
    NppiSize kernel_size = {static_cast<int>(kernel.GetShape()[0]),
                            static_cast<int>(kernel.GetShape()[1])};
    NppiPoint anchor = {static_cast<int>(kernel.GetShape()[0] / 2),
                        static_cast<int>(kernel.GetShape()[1] / 2)};

    // Filter in npp is Convolution, so we need to reverse all the entries.
    core::Tensor kernel_flipped = kernel.Reverse();
    const float *kernel_ptr =
            static_cast<const float *>(kernel_flipped.GetDataPtr());

    auto dtype = src_im.GetDtype();
    auto context = MakeNPPContext();
#define NPP_ARGS                                                          \
    static_cast<const npp_dtype *>(src_im.GetDataPtr()),                  \
            src_im.GetStride(0) * dtype.ByteSize(), src_size, src_offset, \
            static_cast<npp_dtype *>(dst_im.GetDataPtr()),                \
            dst_im.GetStride(0) * dtype.ByteSize(), size_ROI, kernel_ptr, \
            kernel_size, anchor, NPP_BORDER_REPLICATE, context
    if (dtype == core::UInt8) {
        using npp_dtype = Npp8u;
        if (src_im.GetShape(2) == 1) {
            nppiFilterBorder32f_8u_C1R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiFilterBorder32f_8u_C3R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiFilterBorder32f_8u_C4R_Ctx(NPP_ARGS);
        }
    } else if (dtype == core::UInt16) {
        using npp_dtype = Npp16u;
        if (src_im.GetShape(2) == 1) {
            nppiFilterBorder32f_16u_C1R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiFilterBorder32f_16u_C3R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiFilterBorder32f_16u_C4R_Ctx(NPP_ARGS);
        }
    } else if (dtype == core::Float32) {
        using npp_dtype = Npp32f;
        if (src_im.GetShape(2) == 1) {
            nppiFilterBorder_32f_C1R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiFilterBorder_32f_C3R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 4) {
            nppiFilterBorder_32f_C4R_Ctx(NPP_ARGS);
        }
    } else {
        utility::LogError("npp::Filter(): Unsupported dtype {}",
                          dtype.ToString());
    }
#undef NPP_ARGS
}

void FilterBilateral(const core::Tensor &src_im,
                     core::Tensor &dst_im,
                     int kernel_size,
                     float value_sigma,
                     float distance_sigma) {
    if (src_im.GetDevice() != dst_im.GetDevice()) {
        utility::LogError(
                "src_im and dst_im are not on the same device, got {} and {}.",
                src_im.GetDevice().ToString(), dst_im.GetDevice().ToString());
    }
    core::CUDAScopedDevice scoped_device(src_im.GetDevice());

    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.
    NppiSize src_size = {static_cast<int>(src_im.GetShape(1)),
                         static_cast<int>(src_im.GetShape(0))};
    NppiPoint src_offset = {0, 0};

    // create struct with ROI size
    NppiSize size_ROI = {static_cast<int>(dst_im.GetShape(1)),
                         static_cast<int>(dst_im.GetShape(0))};

    auto dtype = src_im.GetDtype();
    auto context = MakeNPPContext();
#define NPP_ARGS                                                               \
    static_cast<const npp_dtype *>(src_im.GetDataPtr()),                       \
            src_im.GetStride(0) * dtype.ByteSize(), src_size, src_offset,      \
            static_cast<npp_dtype *>(dst_im.GetDataPtr()),                     \
            dst_im.GetStride(0) * dtype.ByteSize(), size_ROI, kernel_size / 2, \
            1, value_sigma *value_sigma, distance_sigma *distance_sigma,       \
            NPP_BORDER_REPLICATE, context
    if (dtype == core::UInt8) {
        using npp_dtype = Npp8u;
        if (src_im.GetShape(2) == 1) {
            nppiFilterBilateralGaussBorder_8u_C1R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiFilterBilateralGaussBorder_8u_C3R_Ctx(NPP_ARGS);
        }
    } else if (dtype == core::UInt16) {
        using npp_dtype = Npp16u;
        if (src_im.GetShape(2) == 1) {
            nppiFilterBilateralGaussBorder_16u_C1R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiFilterBilateralGaussBorder_16u_C3R_Ctx(NPP_ARGS);
        }
    } else if (dtype == core::Float32) {
        using npp_dtype = Npp32f;
        if (src_im.GetShape(2) == 1) {
            nppiFilterBilateralGaussBorder_32f_C1R_Ctx(NPP_ARGS);
        } else if (src_im.GetShape(2) == 3) {
            nppiFilterBilateralGaussBorder_32f_C3R_Ctx(NPP_ARGS);
        }
    } else {
        utility::LogError("npp::Filter(): Unsupported dtype {}",
                          dtype.ToString());
    }
#undef NPP_ARGS
}

void FilterGaussian(const core::Tensor &src_im,
                    core::Tensor &dst_im,
                    int kernel_size,
                    float sigma) {
    if (src_im.GetDevice() != dst_im.GetDevice()) {
        utility::LogError(
                "src_im and dst_im are not on the same device, got {} and {}.",
                src_im.GetDevice().ToString(), dst_im.GetDevice().ToString());
    }
    core::CUDAScopedDevice scoped_device(src_im.GetDevice());

    // Generate separable kernel weights given the sigma value.
    core::Tensor dist =
            core::Tensor::Arange(static_cast<float>(-kernel_size / 2),
                                 static_cast<float>(kernel_size / 2 + 1), 1.0f,
                                 core::Float32, src_im.GetDevice());
    core::Tensor logval = (dist * dist).Mul(-0.5f / (sigma * sigma));
    core::Tensor mask = logval.Exp();
    mask = mask / mask.Sum({0});
    mask = mask.View({kernel_size, 1});

    // Use the general Filter, as NPP Gaussian/GaussianAdvanced all return
    // inconsistent results.
    // Outer product
    core::Tensor kernel = mask.Matmul(mask.T()).Contiguous();
    return Filter(src_im, dst_im, kernel);
}

void FilterSobel(const core::Tensor &src_im,
                 core::Tensor &dst_im_dx,
                 core::Tensor &dst_im_dy,
                 int kernel_size) {
    if (src_im.GetDevice() != dst_im_dx.GetDevice() ||
        src_im.GetDevice() != dst_im_dy.GetDevice()) {
        utility::LogError(
                "src_im, dst_im_dx, and dst_im_dy are not on the same device, "
                "got {}, {} and {}.",
                src_im.GetDevice().ToString(), dst_im_dx.GetDevice().ToString(),
                dst_im_dy.GetDevice().ToString());
    }
    core::CUDAScopedDevice scoped_device(src_im.GetDevice());

    // Supported device and datatype checking happens in calling code and will
    // result in an exception if there are errors.
    NppiSize src_size = {static_cast<int>(src_im.GetShape(1)),
                         static_cast<int>(src_im.GetShape(0))};
    NppiPoint src_offset = {0, 0};

    // create struct with ROI size
    NppiSize size_ROI = {static_cast<int>(dst_im_dx.GetShape(1)),
                         static_cast<int>(dst_im_dx.GetShape(0))};
    auto dtype = src_im.GetDtype();
    const static std::unordered_map<int, NppiMaskSize> kernel_size_dict = {
            {3, NPP_MASK_SIZE_3_X_3},
            {5, NPP_MASK_SIZE_5_X_5},
    };
    auto it = kernel_size_dict.find(kernel_size);
    if (it == kernel_size_dict.end()) {
        utility::LogError("Unsupported size {} for NPP FilterSobel",
                          kernel_size);
    }

    // Counterintuitive conventions: dy: Horizontal,  dx: Vertical.
    // Probable reason: dy detects horizontal edges, dx detects vertical edges.
    auto context = MakeNPPContext();
#define NPP_ARGS_DX                                                       \
    static_cast<const npp_src_dtype *>(src_im.GetDataPtr()),              \
            src_im.GetStride(0) * dtype.ByteSize(), src_size, src_offset, \
            static_cast<npp_dst_dtype *>(dst_im_dx.GetDataPtr()),         \
            dst_im_dx.GetStride(0) * dst_im_dx.GetDtype().ByteSize(),     \
            size_ROI, it->second, NPP_BORDER_REPLICATE, context
#define NPP_ARGS_DY                                                       \
    static_cast<const npp_src_dtype *>(src_im.GetDataPtr()),              \
            src_im.GetStride(0) * dtype.ByteSize(), src_size, src_offset, \
            static_cast<npp_dst_dtype *>(dst_im_dy.GetDataPtr()),         \
            dst_im_dy.GetStride(0) * dst_im_dy.GetDtype().ByteSize(),     \
            size_ROI, it->second, NPP_BORDER_REPLICATE, context
    if (dtype == core::UInt8) {
        using npp_src_dtype = Npp8u;
        using npp_dst_dtype = Npp16s;
        nppiFilterSobelVertBorder_8u16s_C1R_Ctx(NPP_ARGS_DX);
        nppiFilterSobelHorizBorder_8u16s_C1R_Ctx(NPP_ARGS_DY);
    } else if (dtype == core::Float32) {
        using npp_src_dtype = Npp32f;
        using npp_dst_dtype = Npp32f;
        nppiFilterSobelVertMaskBorder_32f_C1R_Ctx(NPP_ARGS_DX);
        nppiFilterSobelHorizMaskBorder_32f_C1R_Ctx(NPP_ARGS_DY);
    } else {
        utility::LogError("npp::FilterSobel(): Unsupported dtype {}",
                          dtype.ToString());
    }
#undef NPP_ARGS_DX
#undef NPP_ARGS_DY

    // NPP uses a "right minus left" kernel in 10.2.
    // https://docs.nvidia.com/cuda/npp/group__image__filter__sobel__border.html
    // But it is observed to use "left minus right" in unit tests in 10.1.
    // We need to negate it in-place for lower versions.
    // TODO: this part is subject to changes given tests on more versions.
    int cuda_version;
    OPEN3D_CUDA_CHECK(cudaRuntimeGetVersion(&cuda_version));
    if (cuda_version < 10020) {
        dst_im_dx.Neg_();
    }
}
}  // namespace npp
}  // namespace geometry
}  // namespace t
}  // namespace open3d
