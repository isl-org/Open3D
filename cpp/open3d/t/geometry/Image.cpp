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

#include "open3d/t/geometry/Image.h"

#include <cmath>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/GeometryIndexer.h"
#include "open3d/t/geometry/kernel/IPPImage.h"
#include "open3d/t/geometry/kernel/Image.h"
#include "open3d/t/geometry/kernel/NPPImage.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/utility/Random.h"

namespace open3d {
namespace t {
namespace geometry {

using dtype_channels_pairs = std::vector<std::pair<core::Dtype, int64_t>>;

Image::Image(int64_t rows,
             int64_t cols,
             int64_t channels,
             core::Dtype dtype,
             const core::Device &device)
    : Geometry(Geometry::GeometryType::Image, 2) {
    Reset(rows, cols, channels, dtype, device);
}

Image::Image(const core::Tensor &tensor)
    : Geometry(Geometry::GeometryType::Image, 2) {
    if (!tensor.IsContiguous()) {
        utility::LogError("Input tensor must be contiguous.");
    }
    if (tensor.NumDims() == 2) {
        data_ = tensor.Reshape(
                core::shape_util::Concat(tensor.GetShape(), {1}));
    } else if (tensor.NumDims() == 3) {
        data_ = tensor;
    } else {
        utility::LogError("Input tensor must be 2-D or 3-D, but got shape {}.",
                          tensor.GetShape().ToString());
    }
}

Image &Image::Reset(int64_t rows,
                    int64_t cols,
                    int64_t channels,
                    core::Dtype dtype,
                    const core::Device &device) {
    if (rows < 0) {
        utility::LogError("rows must be >= 0, but got {}.", rows);
    }
    if (cols < 0) {
        utility::LogError("cols must be >= 0, but got {}.", cols);
    }
    if (channels <= 0) {
        utility::LogError("channels must be > 0, but got {}.", channels);
    }

    data_ = core::Tensor({rows, cols, channels}, dtype, device);
    return *this;
}

Image Image::To(core::Dtype dtype,
                bool copy /*= false*/,
                utility::optional<double> scale_ /* = utility::nullopt */,
                double offset /* = 0.0 */) const {
    if (dtype == GetDtype() && !scale_.has_value() && offset == 0.0) {
        return copy ? Image(data_.Clone()) : *this;
    }
    // Check IPP datatype support for each function in IPP documentation:
    // https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing.html
    // IPP supports all pairs of conversions for these data types
    static const std::vector<core::Dtype> ipp_supported{
            {core::Bool},  {core::UInt8},   {core::UInt16},
            {core::Int32}, {core::Float32}, {core::Float64}};

    double scale = 1.0;
    if (!scale_.has_value() &&
        (dtype == core::Float32 || dtype == core::Float64)) {
        if (GetDtype() == core::UInt8) {
            scale = 1. / 255;
        } else if (GetDtype() == core::UInt16) {
            scale = 1. / 65535;
        }
    } else {
        scale = scale_.value_or(1.0);
    }

    Image dst_im;
    if (!copy && dtype == GetDtype()) {
        dst_im.data_ = data_;
    } else {
        dst_im.data_ = core::Tensor::Empty(
                {GetRows(), GetCols(), GetChannels()}, dtype, GetDevice());
    }
    if (HAVE_IPPICV &&  // Check for IPP fast implementation.
        data_.IsCPU() &&
        std::count(ipp_supported.begin(), ipp_supported.end(), GetDtype()) >
                0 &&
        std::count(ipp_supported.begin(), ipp_supported.end(), dtype) > 0) {
        IPP_CALL(ipp::To, data_, dst_im.data_, scale, offset);
    } else {  // NPP does not provide a useful API, so use native kernels
        kernel::image::To(data_, dst_im.data_, scale, offset);
    }
    return dst_im;
}

Image Image::RGBToGray() const {
    if (GetChannels() != 3) {
        utility::LogError(
                "Input image channels must be 3 for RGBToGray, but got {}.",
                GetChannels());
    }
    static const dtype_channels_pairs ipp_supported{
            {core::UInt8, 3},
            {core::UInt16, 3},
            {core::Float32, 3},
    };
    static const dtype_channels_pairs npp_supported{
            {core::UInt8, 3},
            {core::UInt16, 3},
            {core::Float32, 3},
    };

    Image dst_im;
    dst_im.data_ = core::Tensor::Empty({GetRows(), GetCols(), 1}, GetDtype(),
                                       GetDevice());
    if (data_.IsCUDA() &&
        std::count(npp_supported.begin(), npp_supported.end(),
                   std::make_pair(GetDtype(), GetChannels())) > 0) {
        CUDA_CALL(npp::RGBToGray, data_, dst_im.data_);
    } else if (HAVE_IPPICV && data_.IsCPU() &&
               std::count(ipp_supported.begin(), ipp_supported.end(),
                          std::make_pair(GetDtype(), GetChannels())) > 0) {
        IPP_CALL(ipp::RGBToGray, data_, dst_im.data_);
    } else {
        utility::LogError(
                "RGBToGray with data type {} on device {} is not implemented!",
                GetDtype().ToString(), GetDevice().ToString());
    }
    return dst_im;
}

Image Image::Resize(float sampling_rate, InterpType interp_type) const {
    if (sampling_rate == 1.0f) {
        return *this;
    }

    static const dtype_channels_pairs npp_supported{
            {core::UInt8, 1}, {core::UInt16, 1}, {core::Float32, 1},
            {core::UInt8, 3}, {core::UInt16, 3}, {core::Float32, 3},
            {core::UInt8, 4}, {core::UInt16, 4}, {core::Float32, 4},

    };

    static const dtype_channels_pairs ipp_supported{
            {core::UInt8, 1}, {core::UInt16, 1}, {core::Float32, 1},
            {core::UInt8, 3}, {core::UInt16, 3}, {core::Float32, 3},
            {core::UInt8, 4}, {core::UInt16, 4}, {core::Float32, 4},
    };

    Image dst_im;
    dst_im.data_ = core::Tensor::Empty(
            {static_cast<int64_t>(GetRows() * sampling_rate),
             static_cast<int64_t>(GetCols() * sampling_rate), GetChannels()},
            GetDtype(), GetDevice());

    if (data_.IsCUDA() &&
        std::count(npp_supported.begin(), npp_supported.end(),
                   std::make_pair(GetDtype(), GetChannels())) > 0) {
        CUDA_CALL(npp::Resize, data_, dst_im.data_, interp_type);
    } else if (HAVE_IPPICV && data_.IsCPU() &&
               std::count(ipp_supported.begin(), ipp_supported.end(),
                          std::make_pair(GetDtype(), GetChannels())) > 0) {
        IPP_CALL(ipp::Resize, data_, dst_im.data_, interp_type);
    } else {
        utility::LogError(
                "Resize with data type {} on device {} is not "
                "implemented!",
                GetDtype().ToString(), GetDevice().ToString());
    }
    return dst_im;
}

Image Image::Dilate(int kernel_size) const {
    // Check NPP datatype support for each function in documentation:
    // https://docs.nvidia.com/cuda/npp/group__nppi.html
    static const dtype_channels_pairs npp_supported{
            {core::Bool, 1},    {core::UInt8, 1},   {core::UInt16, 1},
            {core::Int32, 1},   {core::Float32, 1}, {core::Bool, 3},
            {core::UInt8, 3},   {core::UInt16, 3},  {core::Int32, 3},
            {core::Float32, 3}, {core::Bool, 4},    {core::UInt8, 4},
            {core::UInt16, 4},  {core::Int32, 4},   {core::Float32, 4},
    };
    // Check IPP datatype support for each function in IPP documentation:
    // https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing.html
    static const dtype_channels_pairs ipp_supported{
            {core::Bool, 1},    {core::UInt8, 1}, {core::UInt16, 1},
            {core::Float32, 1}, {core::Bool, 3},  {core::UInt8, 3},
            {core::Float32, 3}, {core::Bool, 4},  {core::UInt8, 4},
            {core::Float32, 4}};

    Image dst_im;
    dst_im.data_ = core::Tensor::EmptyLike(data_);
    if (data_.IsCUDA() &&
        std::count(npp_supported.begin(), npp_supported.end(),
                   std::make_pair(GetDtype(), GetChannels())) > 0) {
        CUDA_CALL(npp::Dilate, data_, dst_im.data_, kernel_size);
    } else if (HAVE_IPPICV && data_.IsCPU() &&
               std::count(ipp_supported.begin(), ipp_supported.end(),
                          std::make_pair(GetDtype(), GetChannels())) > 0) {
        IPP_CALL(ipp::Dilate, data_, dst_im.data_, kernel_size);
    } else {
        utility::LogError(
                "Dilate with data type {} on device {} is not implemented!",
                GetDtype().ToString(), GetDevice().ToString());
    }
    return dst_im;
}

Image Image::FilterBilateral(int kernel_size,
                             float value_sigma,
                             float dist_sigma) const {
    if (kernel_size < 3) {
        utility::LogError("Kernel size must be >= 3, but got {}.", kernel_size);
    }

    static const dtype_channels_pairs npp_supported{
            {core::UInt8, 1}, {core::UInt16, 1}, {core::Float32, 1},
            {core::UInt8, 3}, {core::UInt16, 3}, {core::Float32, 3},
    };
    static const dtype_channels_pairs ipp_supported{
            {core::UInt8, 1},
            {core::Float32, 1},
            {core::UInt8, 3},
            {core::Float32, 3},
    };

    Image dst_im;
    dst_im.data_ = core::Tensor::EmptyLike(data_);
    if (data_.IsCUDA() &&
        std::count(npp_supported.begin(), npp_supported.end(),
                   std::make_pair(GetDtype(), GetChannels())) > 0) {
        CUDA_CALL(npp::FilterBilateral, data_, dst_im.data_, kernel_size,
                  value_sigma, dist_sigma);
    } else if (HAVE_IPPICV && data_.IsCPU() &&
               std::count(ipp_supported.begin(), ipp_supported.end(),
                          std::make_pair(GetDtype(), GetChannels())) > 0) {
        IPP_CALL(ipp::FilterBilateral, data_, dst_im.data_, kernel_size,
                 value_sigma, dist_sigma);
    } else {
        utility::LogError(
                "FilterBilateral with data type {} on device {} is not "
                "implemented!",
                GetDtype().ToString(), GetDevice().ToString());
    }
    return dst_im;
}

Image Image::Filter(const core::Tensor &kernel) const {
    static const dtype_channels_pairs npp_supported{
            {core::UInt8, 1}, {core::UInt16, 1}, {core::Float32, 1},
            {core::UInt8, 3}, {core::UInt16, 3}, {core::Float32, 3},
            {core::UInt8, 4}, {core::UInt16, 4}, {core::Float32, 4},
    };
    static const dtype_channels_pairs ipp_supported{
            {core::UInt8, 1}, {core::UInt16, 1}, {core::Float32, 1},
            {core::UInt8, 3}, {core::UInt16, 3}, {core::Float32, 3},
            {core::UInt8, 4}, {core::UInt16, 4}, {core::Float32, 4},
    };

    Image dst_im;
    dst_im.data_ = core::Tensor::EmptyLike(data_);
    if (data_.IsCUDA() &&
        std::count(npp_supported.begin(), npp_supported.end(),
                   std::make_pair(GetDtype(), GetChannels())) > 0) {
        CUDA_CALL(npp::Filter, data_, dst_im.data_, kernel);
    } else if (HAVE_IPPICV && data_.IsCPU() &&
               std::count(ipp_supported.begin(), ipp_supported.end(),
                          std::make_pair(GetDtype(), GetChannels())) > 0) {
        IPP_CALL(ipp::Filter, data_, dst_im.data_, kernel);
    } else {
        utility::LogError(
                "Filter with data type {} on device {} is not "
                "implemented!",
                GetDtype().ToString(), GetDevice().ToString());
    }
    return dst_im;
}

Image Image::FilterGaussian(int kernel_size, float sigma) const {
    if (kernel_size < 3 || kernel_size % 2 == 0) {
        utility::LogError("Kernel size must be an odd number >= 3, but got {}.",
                          kernel_size);
    }

    static const dtype_channels_pairs npp_supported{
            {core::UInt8, 1}, {core::UInt16, 1}, {core::Float32, 1},
            {core::UInt8, 3}, {core::UInt16, 3}, {core::Float32, 3},
            {core::UInt8, 4}, {core::UInt16, 4}, {core::Float32, 4},
    };
    static const dtype_channels_pairs ipp_supported{
            {core::UInt8, 1}, {core::UInt16, 1}, {core::Float32, 1},
            {core::UInt8, 3}, {core::UInt16, 3}, {core::Float32, 3},
            {core::UInt8, 4}, {core::UInt16, 4}, {core::Float32, 4},
    };

    Image dst_im;
    dst_im.data_ = core::Tensor::EmptyLike(data_);
    if (data_.IsCUDA() &&
        std::count(npp_supported.begin(), npp_supported.end(),
                   std::make_pair(GetDtype(), GetChannels())) > 0) {
        CUDA_CALL(npp::FilterGaussian, data_, dst_im.data_, kernel_size, sigma);
    } else if (HAVE_IPPICV && data_.IsCPU() &&
               std::count(ipp_supported.begin(), ipp_supported.end(),
                          std::make_pair(GetDtype(), GetChannels())) > 0) {
        IPP_CALL(ipp::FilterGaussian, data_, dst_im.data_, kernel_size, sigma);
    } else {
        utility::LogError(
                "FilterGaussian with data type {} on device {} is not "
                "implemented!",
                GetDtype().ToString(), GetDevice().ToString());
    }
    return dst_im;
}

std::pair<Image, Image> Image::FilterSobel(int kernel_size) const {
    if (!(kernel_size == 3 || kernel_size == 5)) {
        utility::LogError("Kernel size must be 3 or 5, but got {}.",
                          kernel_size);
    }

    // 16 signed is also supported by the engines, but is non-standard thus
    // not supported by open3d. To filter 16 bit unsigned depth images, we
    // recommend first converting to Float32.
    static const dtype_channels_pairs npp_supported{
            {core::UInt8, 1},
            {core::Float32, 1},
    };
    static const dtype_channels_pairs ipp_supported{
            {core::UInt8, 1},
            {core::Float32, 1},
    };

    // Routines: 8u16s, 32f
    Image dst_im_dx, dst_im_dy;
    core::Dtype dtype = GetDtype();
    if (dtype == core::Float32) {
        dst_im_dx = core::Tensor::EmptyLike(data_);
        dst_im_dy = core::Tensor::EmptyLike(data_);
    } else if (dtype == core::UInt8) {
        dst_im_dx = core::Tensor::Empty(data_.GetShape(), core::Int16,
                                        data_.GetDevice());
        dst_im_dy = core::Tensor::Empty(data_.GetShape(), core::Int16,
                                        data_.GetDevice());
    }

    if (data_.IsCUDA() &&
        std::count(npp_supported.begin(), npp_supported.end(),
                   std::make_pair(GetDtype(), GetChannels())) > 0) {
        CUDA_CALL(npp::FilterSobel, data_, dst_im_dx.data_, dst_im_dy.data_,
                  kernel_size);
    } else if (HAVE_IPPICV && data_.IsCPU() &&
               std::count(ipp_supported.begin(), ipp_supported.end(),
                          std::make_pair(GetDtype(), GetChannels())) > 0) {
        IPP_CALL(ipp::FilterSobel, data_, dst_im_dx.data_, dst_im_dy.data_,
                 kernel_size);
    } else {
        utility::LogError(
                "FilterSobel with data type {} on device {} is not "
                "implemented!",
                GetDtype().ToString(), GetDevice().ToString());
    }
    return std::make_pair(dst_im_dx, dst_im_dy);
}

Image Image::PyrDown() const {
    Image blur = FilterGaussian(5, 1.0f);
    return blur.Resize(0.5, InterpType::Nearest);
}

Image Image::PyrDownDepth(float diff_threshold, float invalid_fill) const {
    if (GetRows() <= 0 || GetCols() <= 0 || GetChannels() != 1) {
        utility::LogError(
                "Invalid shape, expected a 1 channel image, but got ({}, {}, "
                "{})",
                GetRows(), GetCols(), GetChannels());
    }

    core::AssertTensorDtype(AsTensor(), core::Float32);

    core::Tensor dst_tensor = core::Tensor::Empty(
            {GetRows() / 2, GetCols() / 2, 1}, GetDtype(), GetDevice());
    t::geometry::kernel::image::PyrDownDepth(AsTensor(), dst_tensor,
                                             diff_threshold, invalid_fill);
    return t::geometry::Image(dst_tensor);
}

Image Image::ClipTransform(float scale,
                           float min_value,
                           float max_value,
                           float clip_fill) const {
    if (GetRows() <= 0 || GetCols() <= 0 || GetChannels() != 1) {
        utility::LogError(
                "Invalid shape, expected a 1 channel image, but got ({}, {}, "
                "{})",
                GetRows(), GetCols(), GetChannels());
    }

    core::AssertTensorDtypes(AsTensor(), {core::UInt16, core::Float32});

    if (scale < 0 || min_value < 0 || max_value < 0) {
        utility::LogError(
                "Expected positive scale, min_value, and max_value, but got "
                "{}, {}, and {}",
                scale, min_value, max_value);
    }
    if (!(std::isnan(clip_fill) || std::isinf(clip_fill) || clip_fill == 0)) {
        utility::LogWarning(
                "The clip_fill value {} is not recommended. Please use Inf, "
                "NaN or, 0",
                clip_fill);
    }

    Image dst_im(GetRows(), GetCols(), 1, core::Float32, data_.GetDevice());
    kernel::image::ClipTransform(data_, dst_im.data_, scale, min_value,
                                 max_value, clip_fill);
    return dst_im;
}

Image Image::CreateVertexMap(const core::Tensor &intrinsics,
                             float invalid_fill) {
    if (GetRows() <= 0 || GetCols() <= 0 || GetChannels() != 1) {
        utility::LogError(
                "Invalid shape, expected a 1 channel image, but got ({}, {}, "
                "{})",
                GetRows(), GetCols(), GetChannels());
    }

    core::AssertTensorDtype(AsTensor(), core::Float32);
    core::AssertTensorShape(intrinsics, {3, 3});

    Image dst_im(GetRows(), GetCols(), 3, GetDtype(), GetDevice());
    kernel::image::CreateVertexMap(data_, dst_im.data_, intrinsics,
                                   invalid_fill);
    return dst_im;
}

Image Image::CreateNormalMap(float invalid_fill) {
    if (GetRows() <= 0 || GetCols() <= 0 || GetChannels() != 3) {
        utility::LogError(
                "Invalid shape, expected a 3 channel image, but got ({}, {}, "
                "{})",
                GetRows(), GetCols(), GetChannels());
    }

    core::AssertTensorDtype(AsTensor(), core::Float32);

    Image dst_im(GetRows(), GetCols(), 3, GetDtype(), GetDevice());
    kernel::image::CreateNormalMap(data_, dst_im.data_, invalid_fill);
    return dst_im;
}

Image Image::ColorizeDepth(float scale, float min_value, float max_value) {
    if (GetRows() <= 0 || GetCols() <= 0 || GetChannels() != 1) {
        utility::LogError(
                "Invalid shape, expected a 1 channel image, but got ({}, {}, "
                "{})",
                GetRows(), GetCols(), GetChannels());
    }

    core::AssertTensorDtypes(AsTensor(), {core::UInt16, core::Float32});

    if (scale < 0 || min_value < 0 || max_value < 0 || min_value >= max_value) {
        utility::LogError(
                "Expected positive scale, min_value, and max_value, but got "
                "{}, {}, and {}",
                scale, min_value, max_value);
    }

    Image dst_im(GetRows(), GetCols(), 3, core::UInt8, GetDevice());
    kernel::image::ColorizeDepth(data_, dst_im.data_, scale, min_value,
                                 max_value);
    return dst_im;
}

Image Image::FromLegacy(const open3d::geometry::Image &image_legacy,
                        const core::Device &device) {
    static const std::unordered_map<int, core::Dtype> kBytesToDtypeMap = {
            {1, core::UInt8},
            {2, core::UInt16},
            {4, core::Float32},
    };

    if (image_legacy.IsEmpty()) {
        return Image(0, 0, 1, core::Float32, device);
    }

    auto iter = kBytesToDtypeMap.find(image_legacy.bytes_per_channel_);
    if (iter == kBytesToDtypeMap.end()) {
        utility::LogError("Unsupported image bytes_per_channel ({})",
                          image_legacy.bytes_per_channel_);
    }

    core::Dtype dtype = iter->second;

    Image image(image_legacy.height_, image_legacy.width_,
                image_legacy.num_of_channels_, dtype, device);

    size_t num_bytes = image_legacy.height_ * image_legacy.BytesPerLine();
    core::MemoryManager::MemcpyFromHost(image.data_.GetDataPtr(), device,
                                        image_legacy.data_.data(), num_bytes);
    return image;
}

open3d::geometry::Image Image::ToLegacy() const {
    auto dtype = GetDtype();
    if (!(dtype == core::UInt8 || dtype == core::UInt16 ||
          dtype == core::Float32))
        utility::LogError("Legacy image does not support data type {}.",
                          dtype.ToString());
    if (!data_.IsContiguous()) {
        utility::LogError("Image tensor must be contiguous.");
    }
    open3d::geometry::Image image_legacy;
    image_legacy.Prepare(static_cast<int>(GetCols()),
                         static_cast<int>(GetRows()),
                         static_cast<int>(GetChannels()),
                         static_cast<int>(dtype.ByteSize()));
    size_t num_bytes = image_legacy.height_ * image_legacy.BytesPerLine();
    core::MemoryManager::MemcpyToHost(image_legacy.data_.data(),
                                      data_.GetDataPtr(), data_.GetDevice(),
                                      num_bytes);
    return image_legacy;
}

std::string Image::ToString() const {
    return fmt::format("Image[size={{{},{}}}, channels={}, {}, {}]", GetRows(),
                       GetCols(), GetChannels(), GetDtype().ToString(),
                       GetDevice().ToString());
}

DepthNoiseSimulator::DepthNoiseSimulator(const std::string &noise_model_path) {
    // data = np.loadtxt(fname, comments='%', skiprows=5)
    const char comment_prefix = '%';
    const int skip_first_n_lines = 5;
    utility::filesystem::CFile file;
    if (!file.Open(noise_model_path, "r")) {
        utility::LogError("Read depth model failed: unable to open file: {}",
                          noise_model_path);
    }
    std::vector<float> data;
    const char *line_buffer;
    for (int i = 0; i < skip_first_n_lines; ++i) {
        if (!(line_buffer = file.ReadLine())) {
            utility::LogError(
                    "Read depth model failed: file {} is less than {} lines.",
                    noise_model_path, skip_first_n_lines);
        }
    }
    while ((line_buffer = file.ReadLine())) {
        std::string line(line_buffer);
        line.erase(std::find(line.begin(), line.end(), comment_prefix),
                   line.end());
        if (line.empty()) {
            continue;
        } else {
            std::istringstream iss(line);
            float value;
            while (iss >> value) {
                data.push_back(value);
            }
        }
    }

    model_ = core::Tensor::Zeros({80, 80, 5}, core::Float32,
                                 core::Device("CPU:0"));
    kernel::TArrayIndexer<int> model_indexer(model_, 3);

    for (int y = 0; y < 80; ++y) {
        for (int x = 0; x < 80; ++x) {
            int idx = (y * 80 + x) * 23 + 3;
            bool all_less_than_8000 = true;
            for (int i = 0; i < 5; ++i) {
                if (data[idx + i] >= 8000) {
                    all_less_than_8000 = false;
                    break;
                }
            }
            if (all_less_than_8000) {
                // model_[y, x, :] = 0
                continue;
            } else {
                for (int i = 0; i < 5; ++i) {
                    *model_indexer.GetDataPtr<float>(i, x, y) =
                            data[idx + 15 + i];
                }
            }
        }
    }
}

Image DepthNoiseSimulator::Simulate(const Image &im_src, float depth_scale) {
    // Sanity checks.
    if (im_src.GetDtype() == core::Float32) {
        if (depth_scale != 1.0) {
            utility::LogWarning(
                    "Depth scale is ignored when input depth is float32.");
        }
    } else if (im_src.GetDtype() == core::UInt16) {
        if (depth_scale <= 0.0) {
            utility::LogError("Depth scale must be positive.");
        }
    } else {
        utility::LogError("Unsupported depth image dtype: {}.",
                          im_src.GetDtype().ToString());
    }
    if (im_src.GetChannels() != 1) {
        utility::LogError("Depth image must have 1 channel.");
    }

    core::Tensor im_src_tensor = im_src.AsTensor();
    const core::Device &original_device = im_src_tensor.GetDevice();
    const core::Dtype &original_dtype = im_src_tensor.GetDtype();
    int width = im_src.GetCols();
    int height = im_src.GetRows();

    im_src_tensor = im_src_tensor.To(core::Device("CPU:0")).Contiguous();
    if (original_dtype == core::UInt16) {
        im_src_tensor = im_src_tensor.To(core::Float32) / depth_scale;
    }
    core::Tensor im_dst_tensor = im_src_tensor.Clone();

    utility::random::NormalGenerator<float> gen_coord(0, 0.25);
    utility::random::NormalGenerator<float> gen_depth(0, 0.027778);

    kernel::TArrayIndexer<int> src_indexer(im_src_tensor, 2);
    kernel::TArrayIndexer<int> dst_indexer(im_dst_tensor, 2);
    kernel::TArrayIndexer<int> model_indexer(model_, 3);

    // To match the original implementation, we try to keep the same variable
    // names with reference to the original code. Compared to the original
    // implementation, parallelization is done in im_dst_tensor per-pixel level,
    // instead of per-image level. Check out the original code at:
    // http://redwood-data.org/indoor/data/simdepth.py.
    core::ParallelFor(
            core::Device("CPU:0"), width * height,
            [&] OPEN3D_DEVICE(int workload_idx) {
                // TArrayIndexer has reverted coordinate order, use (c, r).
                int r;
                int c;
                src_indexer.WorkloadToCoord(workload_idx, &c, &r);

                // Pixel shuffle.
                int x, y;
                float x_noise = deterministic_debug_mode_ ? 0 : gen_coord();
                float y_noise = deterministic_debug_mode_ ? 0 : gen_coord();
                x = std::min(std::max(int(round(c + x_noise)), 0), width - 1);
                y = std::min(std::max(int(round(r + y_noise)), 0), height - 1);

                // Down sample.
                float d = *src_indexer.GetDataPtr<float>(x - x % 2, y - y % 2);

                // Distortion.
                int i2 = int((d + 1) / 2);
                int i1 = i2 - 1;
                float a_ = (d - (i1 * 2 + 1)) / 2;
                int x_ = int(x / 8);
                int y_ = int(y / 6);
                float model_val0 = *model_indexer.GetDataPtr<float>(
                        std::min(std::max(i1, 0), 4), x_, y_);
                float model_val1 = *model_indexer.GetDataPtr<float>(
                        std::min(i2, 4), x_, y_);
                float f = (1 - a_) * model_val0 + a_ * model_val1;
                if (f == 0) {
                    d = 0;
                } else {
                    d = d / f;
                }

                // Quantization and high freq noise.
                float dst_d;
                if (d == 0) {
                    dst_d = 0;
                } else {
                    float d_noise = deterministic_debug_mode_ ? 0 : gen_depth();
                    dst_d = 35.130 * 8 / round((35.130 / d + d_noise) * 8);
                }
                *dst_indexer.GetDataPtr<float>(c, r) = dst_d;
            });

    if (original_dtype == core::UInt16) {
        im_dst_tensor = (im_dst_tensor * depth_scale).To(core::UInt16);
    }
    assert(im_dst_tensor.GetDtype() == original_dtype);
    im_dst_tensor = im_dst_tensor.To(original_device);

    return Image(im_dst_tensor);
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
