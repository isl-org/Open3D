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

#include "VisualizerForAlignment.h"

#include <External/tinyfiledialogs/tinyfiledialogs.h>

namespace three {

void VisualizerForAlignment::PrintVisualizerHelp()
{
	Visualizer::PrintVisualizerHelp();
	PrintInfo("  -- Alignment control --\n");
	PrintInfo("    Ctrl + R     : Reset source and target to initial state.\n");
	PrintInfo("    Ctrl + S     : Save current alignment session into a JSON file.\n");
	PrintInfo("    Ctrl + O     : Load current alignment session from a JSON file.\n");
	PrintInfo("    Ctrl + A     : Align point clouds based on manually annotations.\n");
	PrintInfo("    Ctrl + I     : Run ICP refinement.\n");
	PrintInfo("    Ctrl + D     : Run voxel downsample for both source and target.\n");
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
		const char *pattern[1] = {"*.json"};
		switch (key) {
		case GLFW_KEY_R: {
			*source_copy_ptr_ = *alignment_session_.source_ptr_;
			*target_copy_ptr_ = *alignment_session_.target_ptr_;
			ResetViewPoint(true);
			UpdateGeometry();
			return;
		}
		case GLFW_KEY_S: {
			std::string default_alignment = default_directory_ +
					"alignment.json";
			if (use_dialog_) {
				filename = tinyfd_saveFileDialog("Alignment session",
						default_alignment.c_str(), 1, pattern,
						"JSON file (*.json)");
			} else {
				filename = default_alignment.c_str();
			}
			if (filename != NULL) {
				SaveSessionToFile(filename);
			}
			return;
		}
		case GLFW_KEY_O: {
			std::string default_alignment = default_directory_ +
					"alignment.json";
			if (use_dialog_) {
				filename = tinyfd_openFileDialog("Alignment session",
						default_alignment.c_str(), 1, pattern,
						"JSON file (*.json)", 0);
			} else {
				filename = default_alignment.c_str();
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
		case GLFW_KEY_I: {
			if (use_dialog_) {
				char buff[DEFAULT_IO_BUFFER_SIZE];
				sprintf(buff, "%.4f", max_correspondence_distance_);
				const char *str = tinyfd_inputBox("Set voxel size",
						"Set max correspondence distance for ICP (ignored if it is non-positive)",
						buff);
				if (str == NULL) {
					PrintDebug("Dialog closed.\n");
					return;
				} else {
					char *end;
					errno = 0;
					double l = std::strtod(str, &end);
					if (errno == ERANGE && (l == HUGE_VAL || l == -HUGE_VAL)) {
						PrintDebug("Illegal input, use default max correspondence distance.\n");
					} else {
						max_correspondence_distance_ = l;
					}
				}
			}
			if (max_correspondence_distance_ > 0.0) {
				PrintInfo("ICP with max correspondence distance %.4f.\n",
						max_correspondence_distance_);
				auto result = RegistrationICP(*source_copy_ptr_,
						*target_copy_ptr_, max_correspondence_distance_,
						Eigen::Matrix4d::Identity(),
						TransformationEstimationPointToPoint(true),
						ICPConvergenceCriteria(1e-6, 1e-6, 30));
				PrintInfo("Registration finished with fitness %.4f and RMSE %.4f.\n",
						result.fitness_, result.inlier_rmse_);
				if (result.fitness_ > 0.0) {
					transformation_ = result.transformation_ * transformation_;
					PrintTransformation();
					source_copy_ptr_->Transform(result.transformation_);
					UpdateGeometry();
				}
			} else {
				PrintInfo("No ICP performed due to illegal max correspondence distance.\n");
			}
			return;
		}
		case GLFW_KEY_D: {
			if (use_dialog_) {
				char buff[DEFAULT_IO_BUFFER_SIZE];
				sprintf(buff, "%.4f", voxel_size_);
				const char *str = tinyfd_inputBox("Set voxel size",
						"Set voxel size (ignored if it is non-positive)",
						buff);
				if (str == NULL) {
					PrintDebug("Dialog closed.\n");
					return;
				} else {
					char *end;
					errno = 0;
					double l = std::strtod(str, &end);
					if (errno == ERANGE && (l == HUGE_VAL || l == -HUGE_VAL)) {
						PrintDebug("Illegal input, use default voxel size.\n");
					} else {
						voxel_size_ = l;
					}
				}
			}
			if (voxel_size_ > 0.0) {
				PrintInfo("Voxel downsample with voxel size %.4f.\n",
						voxel_size_);
				*source_copy_ptr_ = *VoxelDownSample(*source_copy_ptr_,
						voxel_size_);
				UpdateGeometry();
			} else {
				PrintInfo("No voxel downsample performed due to illegal voxel size.\n");
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
				*source_copy_ptr_ = *polygon_volume->CropPointCloud(
						*source_copy_ptr_);
				ResetViewPoint(true);
				UpdateGeometry();
			}
			return;
		}
		case GLFW_KEY_E: {
			std::string default_alignment = default_directory_ +
					"alignment.json";
			if (use_dialog_) {
				filename = tinyfd_saveFileDialog("Alignment session",
						default_alignment.c_str(), 1, pattern,
						"JSON file (*.json)");
			} else {
				filename = default_alignment.c_str();
			}
			if (filename != NULL) {
				SaveSessionToFile(filename);
				EvaluateAlignmentAndSave(filename);
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
	alignment_session_.max_correspondence_distance_ =
			max_correspondence_distance_;
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
	max_correspondence_distance_ =
			alignment_session_.max_correspondence_distance_;
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
	CorrespondenceSet corres;
	for (size_t i = 0; i < source_idx.size(); i++) {
		corres.push_back(Eigen::Vector2i(source_idx[i], target_idx[i]));
	}
	PrintInfo("Error is %.4f before alignment.\n",
			p2p.ComputeRMSE(*alignment_session_.source_ptr_,
			*alignment_session_.target_ptr_, corres));
	transformation_ = p2p.ComputeTransformation(
			*alignment_session_.source_ptr_,
			*alignment_session_.target_ptr_, corres);
	PrintTransformation();
	*source_copy_ptr_ = *alignment_session_.source_ptr_;
	source_copy_ptr_->Transform(transformation_);
	PrintInfo("Error is %.4f before alignment.\n",
			p2p.ComputeRMSE(*source_copy_ptr_,
			*alignment_session_.target_ptr_, corres));
	return true;
}

void VisualizerForAlignment::PrintTransformation()
{
	PrintInfo("Current transformation is:\n");
	PrintInfo("\t%.6f %.6f %.6f %.6f\n",
			transformation_(0, 0), transformation_(0, 1),
			transformation_(0, 2), transformation_(0, 3));
	PrintInfo("\t%.6f %.6f %.6f %.6f\n",
			transformation_(1, 0), transformation_(1, 1),
			transformation_(1, 2), transformation_(1, 3));
	PrintInfo("\t%.6f %.6f %.6f %.6f\n",
			transformation_(2, 0), transformation_(2, 1),
			transformation_(2, 2), transformation_(2, 3));
	PrintInfo("\t%.6f %.6f %.6f %.6f\n",
			transformation_(3, 0), transformation_(3, 1),
			transformation_(3, 2), transformation_(3, 3));
}

void VisualizerForAlignment::EvaluateAlignmentAndSave(
		const std::string &filename)
{
	// Evaluate source_copy_ptr_ and target_copy_ptr_
	std::string source_filename = filesystem::GetFileNameWithoutExtension(
			filename) + ".source.ply";
	std::string target_filename = filesystem::GetFileNameWithoutExtension(
			filename) + ".target.ply";
	std::string source_binname = filesystem::GetFileNameWithoutExtension(
			filename) + ".source.bin";
	std::string target_binname = filesystem::GetFileNameWithoutExtension(
			filename) + ".target.bin";
	FILE * f;

	WritePointCloud(source_filename, *source_copy_ptr_);
	auto source_dis = ComputePointCloudToPointCloudDistance(
			*source_copy_ptr_, *target_copy_ptr_);
	f = fopen(source_binname.c_str(), "wb");
	fwrite(source_dis.data(), sizeof(double), source_dis.size(), f);
	fclose(f);
	WritePointCloud(target_filename, *target_copy_ptr_);
	auto target_dis = ComputePointCloudToPointCloudDistance(
			*target_copy_ptr_, *source_copy_ptr_);
	f = fopen(target_binname.c_str(), "wb");
	fwrite(target_dis.data(), sizeof(double), target_dis.size(), f);
	fclose(f);
}

}	// namespace three
