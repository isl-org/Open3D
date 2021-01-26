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

#include "open3d/t/geometry/Image.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open3d/core/CUDAUtils.h"
#include "open3d/core/Dtype.h"
#include "open3d/core/ShapeUtil.h"
#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/kernel/IPPImage.h"
#include "open3d/t/geometry/kernel/NPPImage.h"
#include "open3d/utility/Console.h"
#include "open3d/Macro.h"

namespace open3d {
namespace t {
namespace geometry {

Image::Image(int64_t rows,
             int64_t cols,
             int64_t channels,
             core::Dtype dtype,
             const core::Device &device)
    : Geometry(Geometry::GeometryType::Image, 2) {
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

Image Image::To(core::Dtype dtype,
                bool copy /*= false*/,
                utility::optional<double> scale_ /* = utility::nullopt */,
                double offset /* = 0.0 */) const {
    // Check IPP datatype support for each function in IPP documentation:
    // https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing.html
    // IPP supports all pairs of conversions for these data types
    static const std::vector<core::Dtype> ipp_supported{
            {core::Dtype::Bool},    {core::Dtype::UInt8},
            {core::Dtype::UInt16},  {core::Dtype::Int32},
            {core::Dtype::Float32}, {core::Dtype::Float64}};

    [[maybe_unused]] double scale = 1.0;
    if (!scale_.has_value() &&
        (dtype == core::Dtype::Float32 || dtype == core::Dtype::Float64)) {
        if (GetDtype() == core::Dtype::UInt8) {
            scale = 1. / 255;
        } else if (GetDtype() == core::Dtype::UInt16) {
            scale = 1. / 65535;
        }
    } else {
        scale = scale_.value_or(1.0);
    }

    Image dst_im;
    if (HAVE_IPPICV &&
        std::count(ipp_supported.begin(), ipp_supported.end(), GetDtype()) >
                0 &&
        std::count(ipp_supported.begin(), ipp_supported.end(), dtype) > 0) {
        // TODO: Tensor based Op for saturate_cast / LinearTransform
        // NPP does not expose a useful API, so as a workaround, move data to
        // CPU and use IPP.
        auto device = data_.GetDevice();
        if (device.GetType() != core::Device::DeviceType::CPU) {
            core::Tensor data_CPU = data_.To(core::Device("CPU:0"));
            core::Tensor dst_data_CPU =
                    (dtype == GetDtype()
                             ? data_CPU
                             : core::Tensor::Empty(
                                       std::vector<int64_t>{GetRows(),
                                                            GetCols(),
                                                            GetChannels()},
                                       dtype, core::Device("CPU:0")));
            IPP_CALL(ipp::To, data_CPU, dst_data_CPU, scale, offset);
            if (!copy && dtype == GetDtype()) {
                const_cast<core::Tensor &>(data_).CopyFrom(dst_data_CPU);
                dst_im.data_ = data_;
            } else {
                dst_im.data_ = dst_data_CPU.To(device);
            }
        } else {
            if (!copy && dtype == GetDtype()) {
                dst_im.data_ = data_;
            } else {
                dst_im.data_ = core::Tensor::Empty(
                        std::vector<int64_t>{GetRows(), GetCols(),
                                             GetChannels()},
                        dtype, GetDevice());
            }
            IPP_CALL(ipp::To, data_, dst_im.data_, scale, offset);
        }
    } else {
        utility::LogError(
                "Conversion from {} to {} on device {} is not implemented!",
                GetDtype().ToString(), dtype.ToString(),
                GetDevice().ToString());
    }
    return dst_im;
}

Image Image::Dilate(int half_kernel_size) const {
    using supported_t = std::vector<std::pair<core::Dtype, int64_t>>;

    // Check NPP datatype support for each function in documentation:
    // https://docs.nvidia.com/cuda/npp/group__nppi.html
    static const supported_t npp_supported{
            {core::Dtype::Bool, 1},    {core::Dtype::UInt8, 1},
            {core::Dtype::UInt16, 1},  {core::Dtype::Int32, 1},
            {core::Dtype::Float32, 1}, {core::Dtype::Bool, 3},
            {core::Dtype::UInt8, 3},   {core::Dtype::UInt16, 3},
            {core::Dtype::Int32, 3},   {core::Dtype::Float32, 3},
            {core::Dtype::Bool, 4},    {core::Dtype::UInt8, 4},
            {core::Dtype::UInt16, 4},  {core::Dtype::Int32, 4},
            {core::Dtype::Float32, 4},
    };
    // Check IPP datatype support for each function in IPP documentation:
    // https://software.intel.com/content/www/us/en/develop/documentation/ipp-dev-reference/top/volume-2-image-processing.html
    static const supported_t ipp_supported{
            {core::Dtype::Bool, 1},    {core::Dtype::UInt8, 1},
            {core::Dtype::UInt16, 1},  {core::Dtype::Float32, 1},
            {core::Dtype::Bool, 3},    {core::Dtype::UInt8, 3},
            {core::Dtype::Float32, 3}, {core::Dtype::Bool, 4},
            {core::Dtype::UInt8, 4},   {core::Dtype::Float32, 4}};

    Image dst_im;
    dst_im.data_ = core::Tensor::EmptyLike(data_);
    if (data_.GetDevice().GetType() == core::Device::DeviceType::CUDA &&
        std::count(npp_supported.begin(), npp_supported.end(),
                   std::make_pair(GetDtype(), GetChannels())) > 0) {
        CUDA_CALL(npp::Dilate, data_, dst_im.data_, half_kernel_size);
    } else if (HAVE_IPPICV &&
               data_.GetDevice().GetType() == core::Device::DeviceType::CPU &&
               std::count(ipp_supported.begin(), ipp_supported.end(),
                          std::make_pair(GetDtype(), GetChannels())) > 0) {
        IPP_CALL(ipp::Dilate, data_, dst_im.data_, half_kernel_size);
    } else {
        utility::LogError(
                "Dilate with data type {} on device {} is not implemented!",
                GetDtype().ToString(), GetDevice().ToString());
    }
    return dst_im;
}

Image Image::FromLegacyImage(const open3d::geometry::Image &image_legacy,
                             const core::Device &device) {
    static const std::unordered_map<int, core::Dtype> kBytesToDtypeMap = {
            {1, core::Dtype::UInt8},
            {2, core::Dtype::UInt16},
            {4, core::Dtype::Float32},
    };

    if (image_legacy.IsEmpty()) {
        return Image(0, 0, 1, core::Dtype::Float32, device);
    }

    auto iter = kBytesToDtypeMap.find(image_legacy.bytes_per_channel_);
    if (iter == kBytesToDtypeMap.end()) {
        utility::LogError("[Image] unsupported image bytes_per_channel ({})",
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

open3d::geometry::Image Image::ToLegacyImage() const {
    auto dtype = GetDtype();
    if (!(dtype == core::Dtype::UInt8 || dtype == core::Dtype::UInt16 ||
          dtype == core::Dtype::Float32))
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

}  // namespace geometry
}  // namespace t
}  // namespace open3d
