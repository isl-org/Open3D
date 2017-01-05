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

#include "VisualizerForAlignment.h"

#include <External/tinyfiledialogs/tinyfiledialogs.h>

namespace three {

void VisualizerForAlignment::PrintVisualizerHelp()
{
	Visualizer::PrintVisualizerHelp();
	PrintInfo("  -- Alignment control --\n");
	PrintInfo("    Ctrl + S     : Save current alignment session into a JSON file.\n");
	PrintInfo("    Ctrl + O     : Load current alignment session from a JSON file.\n");
	PrintInfo("    Ctrl + A     : Align point clouds based on manually annotations.\n");
	PrintInfo("    Ctrl + R     : Run ICP refinement.\n");
	PrintInfo("    Ctrl + V     : Run voxel downsample for both source and target.\n");
	PrintInfo("    Ctrl + K     : Load a polygon from a JSON file and crop source.\n");
	PrintInfo("    Ctrl + E     : Evaluate error and save to files.\n");
}

bool VisualizerForAlignment::AddSourceAndTarget(
		std::shared_ptr<PointCloud> source, std::shared_ptr<PointCloud> target)
{
	GetRenderOption().point_size_ = 1.0;
	alignment_session_.source_ptr_ = source;
	alignment_session_.target_ptr_ = target;
	source_copy_ptr_ = std::make_shared<PointCloud>();
	target_copy_ptr_ = std::make_shared<PointCloud>();
	*source_copy_ptr_ = *source;
	*target_copy_ptr_ = *target;
	return AddGeometry(source_copy_ptr_) && AddGeometry(target_copy_ptr_);
}

void VisualizerForAlignment::KeyPressCallback(GLFWwindow *window, int key,
		int scancode, int action, int mods)
{
	if (action == GLFW_PRESS && (mods & GLFW_MOD_CONTROL)) {
		const char *filename;
		switch (key) {
		case GLFW_KEY_S: {
			if (use_dialog_) {
				filename = tinyfd_saveFileDialog("Alignment session",
		const char *pattern[1] = {"*.json"};
			} else {
				filename = "alignment.json";
			}
			if (filename != NULL) {
				SaveSessionToFile(filename);
			}
			return;
		}
		case GLFW_KEY_O: {
			if (use_dialog_) {
				filename = tinyfd_openFileDialog("Alignment session",
						"./alignment.json", 1, pattern, "JSON file (*.json)",
						0);
			} else {
				filename = "alignment.json";
			}
			if (filename != NULL) {
				LoadSessionFromFile(filename);
			}
			return;
		}
		case GLFW_KEY_A: {
			if (AlignWithManualAnnotation()) {
				ResetViewPoint(true);
				UpdateGeometry();
			}
			return;
		}
		case GLFW_KEY_K: {
			if (!filesystem::FileExists(polygon_filename_)) {
				if (use_dialog_) {
					polygon_filename_ = tinyfd_openFileDialog(
							"Bounding polygon", "polygon.json", 0, NULL, NULL,
							0);
				} else {
					polygon_filename_ = "polygon.json";
				}
			}
			auto polygon_volume = std::make_shared<SelectionPolygonVolume>();
			if (ReadIJsonConvertible(polygon_filename_, *polygon_volume)) {
				auto cropped = std::make_shared<PointCloud>();
				polygon_volume->CropGeometry(*source_copy_ptr_, *cropped);
				*source_copy_ptr_ = *cropped;
				ResetViewPoint(true);
				UpdateGeometry();
			}
			return;
		}
		}
	}
	Visualizer::KeyPressCallback(window, key, scancode, action, mods);
}

bool VisualizerForAlignment::SaveSessionToFile(const std::string &filename)
{
	alignment_session_.source_indices_ = source_visualizer_.GetPickedPoints();
	alignment_session_.target_indices_ = target_visualizer_.GetPickedPoints();
	alignment_session_.voxel_size_ = voxel_size_;
	alignment_session_.with_scaling_ = with_scaling_;
	alignment_session_.transformation_ = transformation_;
	return WriteIJsonConvertible(filename, alignment_session_);
}

bool VisualizerForAlignment::LoadSessionFromFile(const std::string &filename)
{
	if (ReadIJsonConvertible(filename, alignment_session_) == false) {
		return false;
	}
	source_visualizer_.GetPickedPoints() = alignment_session_.source_indices_;
	target_visualizer_.GetPickedPoints() = alignment_session_.target_indices_;
	voxel_size_ = alignment_session_.voxel_size_;
	with_scaling_ = alignment_session_.with_scaling_;
	transformation_ = alignment_session_.transformation_;
	*source_copy_ptr_ = *alignment_session_.source_ptr_;
	source_copy_ptr_->Transform(transformation_);
	source_visualizer_.UpdateRender();
	target_visualizer_.UpdateRender();
	ResetViewPoint(true);
	return UpdateGeometry();
}

bool VisualizerForAlignment::AlignWithManualAnnotation()
{
	const auto &source_idx = source_visualizer_.GetPickedPoints();
	const auto &target_idx = target_visualizer_.GetPickedPoints();
	if (source_idx.empty() || target_idx.empty() ||
			source_idx.size() != target_idx.size()) {
		PrintWarning("# of picked points mismatch: %d in source, %d in target.\n",
				(int)source_idx.size(), (int)target_idx.size());
		return false;
	}
	TransformationEstimationPointToPoint p2p(with_scaling_);
	TransformationEstimation::CorrespondenceSet corres;
	for (size_t i = 0; i < source_idx.size(); i++) {
		corres.push_back(std::make_pair((int)source_idx[i],
				(int)target_idx[i]));
	}
	PrintInfo("Error is %.4f before alignment.\n",
			p2p.ComputeError(*alignment_session_.source_ptr_,
			*alignment_session_.target_ptr_, corres));
	transformation_ = p2p.ComputeTransformation(
			*alignment_session_.source_ptr_,
			*alignment_session_.target_ptr_, corres);
	PrintInfo("Transformation is:\n");
	PrintInfo("%.6f %.6f %.6f %.6f\n",
			transformation_(0, 0), transformation_(0, 1),
			transformation_(0, 2), transformation_(0, 3));
	PrintInfo("%.6f %.6f %.6f %.6f\n",
			transformation_(1, 0), transformation_(1, 1),
			transformation_(1, 2), transformation_(1, 3));
	PrintInfo("%.6f %.6f %.6f %.6f\n",
			transformation_(2, 0), transformation_(2, 1),
			transformation_(2, 2), transformation_(2, 3));
	PrintInfo("%.6f %.6f %.6f %.6f\n",
			transformation_(3, 0), transformation_(3, 1),
			transformation_(3, 2), transformation_(3, 3));
	*source_copy_ptr_ = *alignment_session_.source_ptr_;
	source_copy_ptr_->Transform(transformation_);
	PrintInfo("Error is %.4f before alignment.\n",
			p2p.ComputeError(*source_copy_ptr_,
			*alignment_session_.target_ptr_, corres));
	return true;
}

}	// namespace three
