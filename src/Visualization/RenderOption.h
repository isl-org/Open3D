// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include <Core/Core.h>

namespace three {

class RenderOption
{
public:
	enum LightingOption {
		LIGHTING_DEFAULT = 0,
	};
	
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
	};

	const double POINT_SIZE_MAX = 25.0;
	const double POINT_SIZE_MIN = 1.0;
	const double POINT_SIZE_STEP = 1.0;
	const double POINT_SIZE_DEFAULT = 5.0;
	
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
	};
	
	const Eigen::Vector3d MESH_DEFAULT_COLOR =
			Eigen::Vector3d(0.439216, 0.858824, 0.858824);

	// Image options
	enum ImageStretchOption {
		IMAGE_ORIGINAL_SIZE = 0,
		IMAGE_STRETCH_KEEP_RATIO = 1,
		IMAGE_STRETCH_WITH_WINDOW = 2,
	};

public:
	LightingOption GetLightingOption() const {
		return lighting_option_;
	}

	void SetLightingOption(LightingOption option) {
		lighting_option_ = option;
	}

	void ToggleLightOn() {
		light_on_ = !light_on_;
	}
	
	bool IsLightOn() const {
		return light_on_;
	}
	
	void ToggleInterpolationOption() {
		if (interpolation_option_ == TEXTURE_INTERPOLATION_NEAREST) {
			interpolation_option_ = TEXTURE_INTERPOLATION_LINEAR;
		} else {
			interpolation_option_ = TEXTURE_INTERPOLATION_NEAREST;
		}
	}

	TextureInterpolationOption GetInterpolationOption() const {
		return interpolation_option_;
	}

	// PointCloud options
	double GetPointSize() const {
		return point_size_;
	}

	void ChangePointSize(double change) {
		double new_point_size = point_size_ + change * POINT_SIZE_STEP;
		if (new_point_size >= POINT_SIZE_MIN && 
				new_point_size <= POINT_SIZE_MAX)
		{
			point_size_ = new_point_size;
		}
	}

	PointColorOption GetPointColorOption() const {
		return point_color_option_;
	}

	void SetPointColorOption(PointColorOption option) {
		point_color_option_ = option;
	}

	void TogglePointShowNormal() {
		point_show_normal_ = !point_show_normal_;
	}

	bool IsPointNormalShown() const {
		return point_show_normal_;
	}
	
	// TriangleMesh options
	void ToggleShadingOption() {
		if (mesh_shade_option_ == MESHSHADE_FLATSHADE) {
			mesh_shade_option_ = MESHSHADE_SMOOTHSHADE;
		} else {
			mesh_shade_option_ = MESHSHADE_FLATSHADE;
		}
	}

	MeshShadeOption GetMeshShadeOption() const {
		return mesh_shade_option_;
	}

	void SetMeshShadeOption(MeshShadeOption option) {
		mesh_shade_option_ = option;
	}

	MeshColorOption GetMeshColorOption() const {
		return mesh_color_option_;
	}

	void SetMeshColorOption(MeshColorOption option) {
		mesh_color_option_ = option;
	}

	void ToggleMeshShowBackFace() {
		mesh_show_back_face_ = !mesh_show_back_face_;
	}

	bool IsMeshBackFaceShown() const {
		return mesh_show_back_face_;
	}
	
	// Image options
	ImageStretchOption GetImageStretchOption() const {
		return image_stretch_option_;
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

	int GetImageMaxDepth() const { return image_max_depth_; }

	void SetImageMaxDepth(int depth) { image_max_depth_ = depth; }

private:
	// global options
	LightingOption lighting_option_ = LIGHTING_DEFAULT;
	bool light_on_ = false;
	TextureInterpolationOption interpolation_option_ =
			TEXTURE_INTERPOLATION_NEAREST;
	
	// PointCloud options
	double point_size_ = POINT_SIZE_DEFAULT;
	PointColorOption point_color_option_ = POINTCOLOR_DEFAULT;
	bool point_show_normal_ = false;
	
	// TriangleMesh options
	MeshShadeOption mesh_shade_option_ = MESHSHADE_FLATSHADE;
	MeshColorOption mesh_color_option_ = TRIANGLEMESH_DEFAULT;
	bool mesh_show_back_face_ = false;
	
	// Image options
	ImageStretchOption image_stretch_option_ = IMAGE_ORIGINAL_SIZE;
	int image_max_depth_ = 3000;
};

}	// namespace three
