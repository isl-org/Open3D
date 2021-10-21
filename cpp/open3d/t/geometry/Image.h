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

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "open3d/core/Dtype.h"
#include "open3d/core/Tensor.h"
#include "open3d/core/kernel/UnaryEW.h"
#include "open3d/geometry/Image.h"
#include "open3d/t/geometry/Geometry.h"

namespace open3d {
namespace t {
namespace geometry {

/// \class Image
///
/// \brief The Image class stores image with customizable rows, cols, channels,
/// dtype and device.
class Image : public Geometry {
public:
    /// \brief Constructor for image.
    ///
    /// Row-major storage is used, similar to OpenCV. Use (row, col, channel)
    /// indexing order for image creation and accessing. In general, (r, c, ch)
    /// are the preferred variable names for consistency, and avoid using width,
    /// height, u, v, x, y for coordinates.
    ///
    /// \param rows Number of rows of the image, i.e. image height. \p rows must
    /// be non-negative.
    /// \param cols Number of columns of the image, i.e. image width. \p cols
    /// must be non-negative.
    /// \param channels Number of channels of the image. E.g. for RGB image,
    /// channels == 3; for grayscale image, channels == 1. \p channels must be
    /// greater than 0.
    /// \param dtype Data type of the image.
    /// \param device Device where the image is stored.
    Image(int64_t rows = 0,
          int64_t cols = 0,
          int64_t channels = 1,
          core::Dtype dtype = core::Float32,
          const core::Device &device = core::Device("CPU:0"));

    /// \brief Construct from a tensor. The tensor won't be copied and memory
    /// will be shared.
    ///
    /// \param tensor: Tensor of the image. The tensor must be contiguous. The
    /// tensor must be 2D (rows, cols) or 3D (rows, cols, channels).
    Image(const core::Tensor &tensor);

    virtual ~Image() override {}

public:
    /// \brief Clear image contents by resetting the rows and cols to 0, while
    /// keeping channels, dtype and device unchanged.
    Image &Clear() override {
        data_ = core::Tensor({0, 0, GetChannels()}, GetDtype(), GetDevice());
        return *this;
    }

    /// \brief Returns true if rows * cols * channels == 0.
    bool IsEmpty() const override {
        return GetRows() * GetCols() * GetChannels() == 0;
    }

    /// \brief Reinitialize image with new parameters.
    Image &Reset(int64_t rows = 0,
                 int64_t cols = 0,
                 int64_t channels = 1,
                 core::Dtype dtype = core::Float32,
                 const core::Device &device = core::Device("CPU:0"));

public:
    /// \brief Get the number of rows of the image.
    int64_t GetRows() const { return data_.GetShape()[0]; }

    /// \brief Get the number of columns of the image.
    int64_t GetCols() const { return data_.GetShape()[1]; }

    /// \brief Get the number of channels of the image.
    int64_t GetChannels() const { return data_.GetShape()[2]; }

    /// \brief Get dtype of the image.
    core::Dtype GetDtype() const { return data_.GetDtype(); }

    /// \brief Get device of the image.
    core::Device GetDevice() const { return data_.GetDevice(); }

    /// \brief Get pixel(s) in the image.
    ///
    /// If channels == 1, returns a tensor with shape {}, otherwise returns a
    /// tensor with shape {channels,}. The returned tensor is a slice of the
    /// image's tensor, so when modifying the slice, the original tensor will
    /// also be modified.
    core::Tensor At(int64_t r, int64_t c) const {
        if (GetChannels() == 1) {
            return data_[r][c][0];
        } else {
            return data_[r][c];
        }
    }

    /// \brief Get pixel(s) in the image. Returns a tensor with shape {}.
    core::Tensor At(int64_t r, int64_t c, int64_t ch) const {
        return data_[r][c][ch];
    }

    /// \brief Get raw buffer of the Image data.
    void *GetDataPtr() { return data_.GetDataPtr(); }

    /// \brief Get raw buffer of the Image data.
    const void *GetDataPtr() const { return data_.GetDataPtr(); }

    /// \brief Returns the underlying Tensor of the Image.
    core::Tensor AsTensor() const { return data_; }

    /// \brief Transfer the image to a specified device.
    ///
    /// \param device The targeted device to convert to.
    /// \param copy If true, a new image is always created; if false, the
    /// copy is avoided when the original image is already on the targeted
    /// device.
    Image To(const core::Device &device, bool copy = false) const {
        return Image(data_.To(device, copy));
    }

    /// \brief Returns copy of the image on the same device.
    Image Clone() const { return To(GetDevice(), /*copy=*/true); }

    /// \brief Returns an Image with the specified \p dtype.
    ///
    /// \param dtype The targeted dtype to convert to.
    /// \param copy If true, a new tensor is always created; if false, the copy
    /// is avoided when the original tensor already has the targeted dtype.
    /// \param scale Optional scale value. This is 1./255 for UInt8 ->
    /// Float{32,64}, 1./65535 for UInt16 -> Float{32,64} and 1 otherwise
    /// \param offset Optional shift value. Default 0.
    Image To(core::Dtype dtype,
             bool copy = false,
             utility::optional<double> scale = utility::nullopt,
             double offset = 0.0) const;

    /// \brief Function to linearly transform pixel intensities in place.
    ///
    /// \f$image = scale * image + offset\f$.
    ///
    /// \param scale First multiply image pixel values with this factor. This
    /// should be positive for unsigned dtypes.
    /// \param offset Then add this factor to all image pixel values.
    ///
    /// \return Reference to self.
    Image &LinearTransform(double scale = 1.0, double offset = 0.0) {
        To(GetDtype(), false, scale, offset);
        return *this;
    }

    /// \brief Converts a 3-channel RGB image to a new 1-channel Grayscale image
    ///
    /// Uses formula \f$I = 0.299 * R + 0.587 * G + 0.114 * B\f$.
    Image RGBToGray() const;

    /// Image interpolation algorithms.
    enum class InterpType {
        Nearest = 0,  ///< Nearest neighbors interpolation.
        Linear = 1,   ///< Bilinear interpolation.
        Cubic = 2,    ///< Bicubic interpolation.
        Lanczos = 3,  ///< Lanczos filter interpolation.
        Super = 4     ///< Super sampling interpolation (only downsample).
    };
    /// \brief Return a new image after resizing with specified interpolation
    /// type.
    ///
    /// Downsample if sampling rate is < 1. Upsample if sampling rate > 1.
    /// Aspect ratio is always preserved.
    Image Resize(float sampling_rate = 0.5f,
                 InterpType interp_type = InterpType::Nearest) const;

    /// \brief Return a new image after performing morphological dilation.
    ///
    /// Supported datatypes are UInt8, UInt16 and Float32 with {1, 3, 4}
    /// channels. An 8-connected neighborhood is used to create the dilation
    /// mask.
    ///
    /// \param kernel_size An odd number >= 3.
    Image Dilate(int kernel_size = 3) const;

    /// \brief Return a new image after filtering with the given kernel.
    Image Filter(const core::Tensor &kernel) const;

    /// \brief Return a new image after bilateral filtering.
    ///
    /// \param value_sigma Standard deviation for the image content.
    /// \param distance_sigma Standard deviation for the image pixel positions.
    ///
    /// Note: CPU (IPP) and CUDA (NPP) versions use different algorithms and
    /// will give different results:\n
    /// CPU uses a round kernel (radius = floor(kernel_size / 2)),\n
    /// while CUDA uses a square kernel (width = kernel_size).\n
    /// Make sure to tune parameters accordingly.
    Image FilterBilateral(int kernel_size = 3,
                          float value_sigma = 20.0f,
                          float distance_sigma = 10.0f) const;

    /// \brief Return a new image after Gaussian filtering.
    ///
    /// \param kernel_size Odd numbers >= 3 are supported.
    /// \param sigma Standard deviation of the Gaussian distribution.
    Image FilterGaussian(int kernel_size = 3, float sigma = 1.0f) const;

    /// \brief Return a pair of new gradient images (dx, dy) after Sobel
    /// filtering.
    ///
    /// \param kernel_size: Sobel filter kernel size, either 3 or 5.
    std::pair<Image, Image> FilterSobel(int kernel_size = 3) const;

    /// \brief Return a new downsampled image with pyramid downsampling.
    ///
    /// The returned image is formed by a chained Gaussian filter (kernel_size =
    /// 5, sigma = 1.0) and a resize (ratio = 0.5) operation.
    ///
    /// \returns Half sized downsampled depth image.
    Image PyrDown() const;

    /// \brief Edge and invalid value preserving downsampling by 2 specifically
    /// for depth images.
    ///
    /// Only 1 channel Float32 images are supported. The returned image is
    /// formed by a chained Gaussian filter (kernel_size = 5, sigma = 1.0) and a
    /// resize (ratio = 0.5) operation.
    ///
    /// \param diff_threshold The Gaussian filter averaging ignores neighboring
    /// values if the depth difference is larger than this value.
    /// \param invalid_fill The Gaussian filter ignores these values (may be
    /// specified as NAN, INFINITY or 0.0 (default)).
    ///
    /// \returns Half sized downsampled Float32 depth image.
    Image PyrDownDepth(float diff_threshold, float invalid_fill = 0.f) const;

    /// \brief Return new image after scaling and clipping image values.
    ///
    /// This is typically used for preprocessing a depth image. Images of shape
    /// (rows, cols, channels=1) and Dtypes UInt16 and Float32 are supported.
    /// Each pixel will be transformed by
    /// - x = x / \p scale
    /// - x = x < \p min_value ? \p clip_fill : x
    /// - x = x > \p max_value ? \p clip_fill : x
    ///
    /// Use INFINITY, NAN or 0.0 (default) for \p clip_fill.
    /// \return Transformed image of type Float32, with out-of-range pixels
    /// clipped and assigned the \p clip_fill value.
    Image ClipTransform(float scale,
                        float min_value,
                        float max_value,
                        float clip_fill = 0.0f) const;

    /// \brief Create a vertex map from a depth image using unprojection.
    ///
    /// The input depth (of shape (rows, cols, channels=1) and Dtype Float32) is
    /// expected to be the output of ClipTransform.
    ///
    /// \param intrinsics Pinhole camera model of (3, 3) in Float64.
    /// \param invalid_fill Value to fill in for invalid depths. Use NAN,
    /// INFINITY or 0.0 (default). Must be consistent with \p clip_fill in
    /// ClipTransform.
    ///
    /// \returns Vertex map of shape (rows, cols, channels=3) and Dtype Float32,
    /// with invalid points assigned the \p invalid_fill value.
    Image CreateVertexMap(const core::Tensor &intrinsics,
                          float invalid_fill = 0.0f);

    /// \brief Create a normal map from a vertex map.
    ///
    /// The input vertex map image should be of shape (rows, cols, channels=3)
    /// and Dtype Float32.  This uses a cross product of \f$V(r, c+1)-V(r, c)\f$
    /// and \f$V(r+1, c)-V(r, c)\f$. The input vertex map is expected to be the
    /// output of CreateVertexMap. You may need to start with a filtered depth
    /// image (e.g. with FilterBilateral) to obtain good results.
    ///
    /// \param invalid_fill Value to fill in for invalid points, and to fill-in
    /// if no valid neighbor is found. Use NAN, INFINITY or 0.0 (default). Must
    /// be consistent with \p clip_fill in CreateVertexMap.
    ///
    /// \returns Normal map of shape (rows, cols, channels=3) and Dtype Float32,
    /// with invalid normals assigned the \p invalid_fill value.
    Image CreateNormalMap(float invalid_fill = 0.0f);

    /// \brief Colorize an input depth image (with Dtype UInt16 or Float32).
    ///
    /// The image values are divided by scale, then clamped within [min_value,
    /// max_value] and finally converted to an RGB image using the Turbo
    /// colormap as a lookup table.
    ///
    /// \returns Full color depth map of shape (rows, cols, channels=3) and
    /// Dtype UInt8.
    Image ColorizeDepth(float scale, float min_value, float max_value);

    /// \brief Compute min 2D coordinates for the data (always {0, 0}).
    core::Tensor GetMinBound() const {
        return core::Tensor::Zeros({2}, core::Int64);
    }

    /// \brief Compute max 2D coordinates for the data ({rows, cols}).
    core::Tensor GetMaxBound() const {
        return core::Tensor(std::vector<int64_t>{GetRows(), GetCols()}, {2},
                            core::Int64);
    }

    /// \brief Create from a legacy Open3D Image.
    static Image FromLegacy(const open3d::geometry::Image &image_legacy,
                            const core::Device &Device = core::Device("CPU:0"));

    /// \brief Convert to legacy Image type.
    open3d::geometry::Image ToLegacy() const;

    /// \brief Text description.
    std::string ToString() const;

    /// Do we use IPP ICV for accelerating image processing operations?
#ifdef WITH_IPPICV
    static constexpr bool HAVE_IPPICV = true;
#else
    static constexpr bool HAVE_IPPICV = false;
#endif

protected:
    /// Internal data of the Image, represented as a contiguous 3D tensor of
    /// shape {rows, cols, channels}. Image properties can be obtained from the
    /// tensor.
    core::Tensor data_;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
