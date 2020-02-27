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

#pragma once

#include <Eigen/Core>

#include "Open3D/Utility/IJsonConvertible.h"

namespace open3d {
namespace visualization {

/// \class RenderOption
///
/// \brief Defines rendering options for visualizer.
class RenderOption : public utility::IJsonConvertible {
public:
    // Global options
    enum class TextureInterpolationOption {
        Nearest = 0,
        Linear = 1,
    };

    enum class DepthFunc {
        Never = 0,
        Less = 1,
        Equal = 2,
        LEqual = 3,
        Greater = 4,
        NotEqual = 5,
        GEqual = 6,
        Always = 7
    };

    /// \enum PointColorOption
    ///
    /// \brief Enum class for point color for PointCloud.
    enum class PointColorOption {
        Default = 0,
        Color = 1,
        XCoordinate = 2,
        YCoordinate = 3,
        ZCoordinate = 4,
        Normal = 9,
    };

    const double POINT_SIZE_MAX = 25.0;
    const double POINT_SIZE_MIN = 1.0;
    const double POINT_SIZE_STEP = 1.0;
    const double POINT_SIZE_DEFAULT = 5.0;
    const double LINE_WIDTH_MAX = 10.0;
    const double LINE_WIDTH_MIN = 1.0;
    const double LINE_WIDTH_STEP = 1.0;
    const double LINE_WIDTH_DEFAULT = 1.0;

    /// \enum MeshShadeOption
    ///
    /// \brief Enum class for mesh shading for TriangleMesh.
    enum class MeshShadeOption {
        FlatShade = 0,
        SmoothShade = 1,
    };

    /// \enum MeshColorOption
    ///
    /// \brief Enum class for color for TriangleMesh.
    enum class MeshColorOption {
        Default = 0,
        Color = 1,
        XCoordinate = 2,
        YCoordinate = 3,
        ZCoordinate = 4,
        Normal = 9,
    };

    // Image options
    enum class ImageStretchOption {
        OriginalSize = 0,
        StretchKeepRatio = 1,
        StretchWithWindow = 2,
    };

public:
    /// \brief Default Constructor.
    RenderOption() {
        // VS2013 does not fully support C++11
        // Array initialization has to be done in constructors.
        light_position_relative_[0] = Eigen::Vector3d(0, 0, 2);
        light_position_relative_[1] = Eigen::Vector3d(0, 0, 2);
        light_position_relative_[2] = Eigen::Vector3d(0, 0, -2);
        light_position_relative_[3] = Eigen::Vector3d(0, 0, -2);
        light_color_[0] = Eigen::Vector3d::Ones();
        light_color_[1] = Eigen::Vector3d::Ones();
        light_color_[2] = Eigen::Vector3d::Ones();
        light_color_[3] = Eigen::Vector3d::Ones();
        light_diffuse_power_[0] = 0.66;
        light_diffuse_power_[1] = 0.66;
        light_diffuse_power_[2] = 0.66;
        light_diffuse_power_[3] = 0.66;
        light_specular_power_[0] = 0.2;
        light_specular_power_[1] = 0.2;
        light_specular_power_[2] = 0.2;
        light_specular_power_[3] = 0.2;
        light_specular_shininess_[0] = 100.0;
        light_specular_shininess_[1] = 100.0;
        light_specular_shininess_[2] = 100.0;
        light_specular_shininess_[3] = 100.0;
    }
    ~RenderOption() override {}

public:
    bool ConvertToJsonValue(Json::Value &value) const override;
    bool ConvertFromJsonValue(const Json::Value &value) override;

public:
    void ToggleLightOn() { light_on_ = !light_on_; }
    void ToggleInterpolationOption() {
        if (interpolation_option_ == TextureInterpolationOption::Nearest) {
            interpolation_option_ = TextureInterpolationOption::Linear;
        } else {
            interpolation_option_ = TextureInterpolationOption::Nearest;
        }
    }
    void ChangePointSize(double change) {
        point_size_ = std::max(std::min(point_size_ + change * POINT_SIZE_STEP,
                                        POINT_SIZE_MAX),
                               POINT_SIZE_MIN);
    }
    void SetPointSize(double size) {
        point_size_ = std::max(std::min(size, POINT_SIZE_MAX), POINT_SIZE_MIN);
    }
    void ChangeLineWidth(double change) {
        line_width_ = std::max(std::min(line_width_ + change * LINE_WIDTH_STEP,
                                        LINE_WIDTH_MAX),
                               LINE_WIDTH_MIN);
    }
    void TogglePointShowNormal() { point_show_normal_ = !point_show_normal_; }
    void ToggleShadingOption() {
        if (mesh_shade_option_ == MeshShadeOption::FlatShade) {
            mesh_shade_option_ = MeshShadeOption::SmoothShade;
        } else {
            mesh_shade_option_ = MeshShadeOption::FlatShade;
        }
    }
    void ToggleMeshShowBackFace() {
        mesh_show_back_face_ = !mesh_show_back_face_;
    }
    void ToggleMeshShowWireframe() {
        mesh_show_wireframe_ = !mesh_show_wireframe_;
    }
    void ToggleImageStretchOption() {
        if (image_stretch_option_ == ImageStretchOption::OriginalSize) {
            image_stretch_option_ = ImageStretchOption::StretchKeepRatio;
        } else if (image_stretch_option_ ==
                   ImageStretchOption::StretchKeepRatio) {
            image_stretch_option_ = ImageStretchOption::StretchWithWindow;
        } else {
            image_stretch_option_ = ImageStretchOption::OriginalSize;
        }
    }

    int GetGLDepthFunc() const;

public:
    // global options
    /// Background RGB color.
    Eigen::Vector3d background_color_ = Eigen::Vector3d::Ones();
    TextureInterpolationOption interpolation_option_ =
            TextureInterpolationOption::Nearest;
    DepthFunc depthFunc_ = DepthFunc::Less;

    // Phong lighting options
    /// Whether to turn on Phong lighting.
    bool light_on_ = true;
    Eigen::Vector3d light_position_relative_[4];
    Eigen::Vector3d light_color_[4];
    Eigen::Vector3d light_ambient_color_ = Eigen::Vector3d::Zero();
    double light_diffuse_power_[4];
    double light_specular_power_[4];
    double light_specular_shininess_[4];

    // PointCloud options
    /// Point size for PointCloud.
    double point_size_ = POINT_SIZE_DEFAULT;
    /// Point color option for PointCloud.
    PointColorOption point_color_option_ = PointColorOption::Default;
    /// Whether to show normal for PointCloud.
    bool point_show_normal_ = false;

    // TriangleMesh options
    /// Mesh shading option for TriangleMesh.
    MeshShadeOption mesh_shade_option_ = MeshShadeOption::FlatShade;
    /// Color option for TriangleMesh.
    MeshColorOption mesh_color_option_ = MeshColorOption::Color;
    /// Whether to show back faces for TriangleMesh.
    bool mesh_show_back_face_ = false;

    bool mesh_show_wireframe_ = false;
    Eigen::Vector3d default_mesh_color_ = Eigen::Vector3d(0.7, 0.7, 0.7);

    // LineSet options
    /// Line width for LineSet.
    double line_width_ = LINE_WIDTH_DEFAULT;

    // Image options
    ImageStretchOption image_stretch_option_ =
            ImageStretchOption::StretchKeepRatio;
    int image_max_depth_ = 3000;

    // Coordinate frame
    /// Whether to show coordinate frame.
    bool show_coordinate_frame_ = false;
};

}  // namespace visualization
}  // namespace open3d
