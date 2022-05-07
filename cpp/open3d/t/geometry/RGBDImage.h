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

#include "open3d/geometry/RGBDImage.h"
#include "open3d/t/geometry/Image.h"

namespace open3d {
namespace t {
namespace geometry {

/// \brief RGBDImage A pair of color and depth images.
///
/// For most processing, the image pair should be aligned (same viewpoint and
/// resolution).
class RGBDImage : public Geometry {
public:
    /// \brief Default Comnstructor.
    RGBDImage() : Geometry(Geometry::GeometryType::RGBDImage, 2) {}
    /// \brief Parameterized Constructor.
    ///
    /// \param color The color image.
    /// \param depth The depth image.
    /// \param aligned Are the two images aligned (same viewpoint and
    /// resolution)?
    RGBDImage(const Image &color, const Image &depth, bool aligned = true)
        : Geometry(Geometry::GeometryType::RGBDImage, 2),
          color_(color),
          depth_(depth),
          aligned_(aligned) {
        if (color.GetRows() != depth.GetRows() ||
            color.GetCols() != depth.GetCols()) {
            aligned_ = false;
            utility::LogWarning(
                    "Aligned image pair must have the same resolution.");
        }
    }

    ~RGBDImage() override{};

    /// Clear stored data.
    RGBDImage &Clear() override;

    /// Is any data stored?
    bool IsEmpty() const override;

    /// Are the depth and color images aligned (same viewpoint and resolution)?
    bool AreAligned() const { return aligned_; }

    /// Compute min 2D coordinates for the data (always {0,0}).
    core::Tensor GetMinBound() const {
        return core::Tensor::Zeros({2}, core::Int64);
    };

    /// Compute max 2D coordinates for the data.
    core::Tensor GetMaxBound() const {
        return core::Tensor(
                std::vector<int64_t>{color_.GetCols() + depth_.GetCols(),
                                     color_.GetRows()},
                {2}, core::Int64);
    };

    /// Transfer the RGBD image to a specified device.
    /// \param device The targeted device to convert to.
    /// \param copy If true, a new image is always created; if false, the
    /// copy is avoided when the original image is already on the target
    /// device.
    RGBDImage To(const core::Device &device, bool copy = false) const {
        return RGBDImage(color_.To(device, copy), depth_.To(device, copy),
                         aligned_);
    }

    /// Returns copy of the RGBD image on the same device.
    RGBDImage Clone() const { return To(color_.GetDevice(), /*copy=*/true); }

    /// Convert to the legacy RGBDImage format.
    open3d::geometry::RGBDImage ToLegacy() const {
        return open3d::geometry::RGBDImage(color_.ToLegacy(),
                                           depth_.ToLegacy());
    }

    /// Text description.
    std::string ToString() const;

public:
    /// The color image.
    Image color_;
    /// The depth image.
    Image depth_;
    /// Are the depth and color images aligned (same viewpoint and resolution)?
    bool aligned_ = true;
};

}  // namespace geometry
}  // namespace t
}  // namespace open3d
