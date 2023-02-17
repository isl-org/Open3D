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

#include <string>

#include "open3d/io/ImageIO.h"
#include "open3d/t/geometry/Image.h"

namespace open3d {
namespace t {
namespace io {

/// Factory function to create an image from a file (ImageFactory.cpp)
/// Return an empty image if fail to read the file.
std::shared_ptr<geometry::Image> CreateImageFromFile(
        const std::string &filename);

/// The general entrance for reading an Image from a file
/// The function calls read functions based on the extension name of filename.
/// \param filename Full path to image. Supported file formats are png,
/// jpg/jpeg.
/// \param image An object of type open3d::t::geometry::Image.
/// \return return true if the read function is successful, false otherwise.
bool ReadImage(const std::string &filename, geometry::Image &image);

constexpr int kOpen3DImageIODefaultQuality = -1;

/// The general entrance for writing an Image to a file
/// The function calls write functions based on the extension name of filename.
/// If the write function supports quality, the parameter will be used.
/// Otherwise it will be ignored.
/// \param filename Full path to image. Supported file formats are png,
/// jpg/jpeg.
/// \param image An object of type open3d::t::geometry::Image.
/// \param quality: PNG: [0-9] <=2 fast write for storing intermediate data
///                            >=3 (default) normal write for balanced speed and
///                            file size
///                 JPEG: [0-100] Typically in [70,95]. 90 is default (good
///                 quality).
/// \return return true if the write function is successful, false otherwise.
///
/// Supported file extensions are png, jpg/jpeg. Data type and number of
/// channels depends on the file extension.
/// - PNG: Dtype should be one of core::UInt8, core::UInt16
///        Supported number of channels are 1, 3, and 4.
/// - JPG: Dtyppe should be core::UInt8
///        Supported number of channels are 1 and 3.
bool WriteImage(const std::string &filename,
                const geometry::Image &image,
                int quality = kOpen3DImageIODefaultQuality);

bool ReadImageFromPNG(const std::string &filename, geometry::Image &image);

bool WriteImageToPNG(const std::string &filename,
                     const geometry::Image &image,
                     int quality = kOpen3DImageIODefaultQuality);

bool ReadImageFromJPG(const std::string &filename, geometry::Image &image);

bool WriteImageToJPG(const std::string &filename,
                     const geometry::Image &image,
                     int quality = kOpen3DImageIODefaultQuality);

/// Simulate depth image noise from a given noise distortion model. The
/// distortion model is based on *Teichman et. al. "Unsupervised intrinsic
/// calibration of depth sensors via SLAM" RSS 2009*. Also see
/// [Redwood dataset](http://redwood-data.org/indoor/dataset.html)
class DepthNoiseSimulator {
public:
    /// \brief Constructor.
    /// \param noise_model Path to the noise model file. See
    /// http://redwood-data.org/indoor/dataset.html for the format. Or, you may
    /// use one of our example datasets, e.g., RedwoodIndoorLivingRoom1.
    explicit DepthNoiseSimulator(const std::string &noise_model_path);

    /// \brief Apply noise model to a depth image.
    ///
    /// \param im_src Source depth image, must be with dtype UInt16 or
    /// Float32, channels==1.
    /// \param depth_scale Scale factor to the depth image. As a sanity check,
    /// if the dtype is Float32, the depth_scale must be 1.0. If the dtype is
    /// is UInt16, the depth_scale is typically larger than 1.0, e.g. it can be
    /// 1000.0.
    /// \return Noisy depth image with the same shape and dtype as \p im_src.
    geometry::Image Simulate(const geometry::Image &im_src,
                             float depth_scale = 1000.0);

    /// \brief Return the noise model.
    core::Tensor GetNoiseModel() const { return model_; }

    /// \brief Enable deterministic debug mode. All normally distributed noise
    /// will be replaced by 0.
    void EnableDeterministicDebugMode() { deterministic_debug_mode_ = true; }

private:
    core::Tensor model_;  // ndims==3
    bool deterministic_debug_mode_ = false;
};

}  // namespace io
}  // namespace t
}  // namespace open3d
