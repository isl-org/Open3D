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

#include <IO/ClassIO/IJsonConvertible.h>

namespace three {

class RenderOption : public IJsonConvertible
{
public:
	// Global options
	enum TextureInterpolationOption {
		TEXTURE_INTERPOLATION_NEAREST = 0,
		TEXTURE_INTERPOLATION_LINEAR = 1,
	};
	
	// PointCloud options
	enum PointColorOption {
		POINTCOLOR_DEFAULT = 0,
		POINTCOLOR_COLOR = 1,
		POINTCOLOR_X = 2,
		POINTCOLOR_Y = 3,
		POINTCOLOR_Z = 4,
		POINTCOLOR_NORMAL = 9,
	};

	static const double POINT_SIZE_MAX;
	static const double POINT_SIZE_MIN;
	static const double POINT_SIZE_STEP;
	static const double POINT_SIZE_DEFAULT;
	
	// TriangleMesh options
	enum MeshShadeOption {
		MESHSHADE_FLATSHADE = 0,
		MESHSHADE_SMOOTHSHADE = 1,
	};

	enum MeshColorOption {
		TRIANGLEMESH_DEFAULT = 0,
		TRIANGLEMESH_COLOR = 1,
		TRIANGLEMESH_X = 2,
		TRIANGLEMESH_Y = 3,
		TRIANGLEMESH_Z = 4,
		TRIANGLEMESH_NORMAL = 9,
	};
	
	// Image options
	enum ImageStretchOption {
		IMAGE_ORIGINAL_SIZE = 0,
		IMAGE_STRETCH_KEEP_RATIO = 1,
		IMAGE_STRETCH_WITH_WINDOW = 2,
	};

public:
	RenderOption() {
		// VS2013 does not fully support C++11
		// Array initialization has to be done in contructors.
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
	void ToggleLightOn() {
		light_on_ = !light_on_;
	}
	void ToggleInterpolationOption() {
		if (interpolation_option_ == TEXTURE_INTERPOLATION_NEAREST) {
			interpolation_option_ = TEXTURE_INTERPOLATION_LINEAR;
		} else {
			interpolation_option_ = TEXTURE_INTERPOLATION_NEAREST;
		}
	}
	void ChangePointSize(double change) {
		point_size_ = std::max(std::min(point_size_ + change * POINT_SIZE_STEP, 
				POINT_SIZE_MAX), POINT_SIZE_MIN);
	}
	void TogglePointShowNormal() {
		point_show_normal_ = !point_show_normal_;
	}
	void ToggleShadingOption() {
		if (mesh_shade_option_ == MESHSHADE_FLATSHADE) {
			mesh_shade_option_ = MESHSHADE_SMOOTHSHADE;
		} else {
			mesh_shade_option_ = MESHSHADE_FLATSHADE;
		}
	}
	void ToggleMeshShowBackFace() {
		mesh_show_back_face_ = !mesh_show_back_face_;
	}
	void ToggleMeshShowWireframe() {
		mesh_show_wireframe_ = !mesh_show_wireframe_;
	}
	void ToggleImageStretchOption() {
		if (image_stretch_option_ == IMAGE_ORIGINAL_SIZE) {
			image_stretch_option_ = IMAGE_STRETCH_KEEP_RATIO;
		} else if (image_stretch_option_ == IMAGE_STRETCH_KEEP_RATIO) {
			image_stretch_option_ = IMAGE_STRETCH_WITH_WINDOW;
		} else {
			image_stretch_option_ = IMAGE_ORIGINAL_SIZE;
		}
	}

public:
	// global options
	Eigen::Vector3d background_color_ = Eigen::Vector3d::Ones();
	TextureInterpolationOption interpolation_option_ =
			TEXTURE_INTERPOLATION_NEAREST;

	// Phong lighting options
	bool light_on_ = true;
	Eigen::Vector3d light_position_relative_[4];
	Eigen::Vector3d light_color_[4];
	Eigen::Vector3d light_ambient_color_ = Eigen::Vector3d::Zero();
	double light_diffuse_power_[4];
	double light_specular_power_[4];
	double light_specular_shininess_[4];

	// PointCloud options
	double point_size_ = POINT_SIZE_DEFAULT;
	PointColorOption point_color_option_ = POINTCOLOR_DEFAULT;
	bool point_show_normal_ = false;
	
	// TriangleMesh options
	MeshShadeOption mesh_shade_option_ = MESHSHADE_FLATSHADE;
	MeshColorOption mesh_color_option_ = TRIANGLEMESH_COLOR;
	bool mesh_show_back_face_ = false;
	bool mesh_show_wireframe_ = false;
	Eigen::Vector3d default_mesh_color_ = Eigen::Vector3d(
			0.7, 0.7, 0.7);
	
	// Image options
	ImageStretchOption image_stretch_option_ = IMAGE_ORIGINAL_SIZE;
	int image_max_depth_ = 3000;

	// Coordinate frame
	bool show_coordinate_frame_ = false;
};

}	// namespace three
