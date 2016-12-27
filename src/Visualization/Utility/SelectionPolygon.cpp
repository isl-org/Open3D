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

#include "SelectionPolygon.h"

#include <Core/Geometry/PointCloud.h>
#include <Visualization/Visualizer/ViewControl.h>

namespace three{

void SelectionPolygon::Clear()
{
	polygon_.clear();
	is_closed_ = false;
	polygon_interior_mask_.Clear();
	polygon_type_ = POLYGON_UNFILLED;
}

bool SelectionPolygon::IsEmpty() const
{
	// A valid polygon, either close or open, should have at least 2 vertices.
	return polygon_.size() <= 1;
}

Eigen::Vector2d SelectionPolygon::GetMinBound() const
{
	if (polygon_.empty()) {
		return Eigen::Vector2d(0.0, 0.0);
	}
	auto itr_x = std::min_element(polygon_.begin(), polygon_.end(),
		[](Eigen::Vector2d a, Eigen::Vector2d b) { return a(0) < b(0); });
	auto itr_y = std::min_element(polygon_.begin(), polygon_.end(),
		[](Eigen::Vector2d a, Eigen::Vector2d b) { return a(1) < b(1); });
	return Eigen::Vector2d((*itr_x)(0), (*itr_y)(1));
}

Eigen::Vector2d SelectionPolygon::GetMaxBound() const
{
	if (polygon_.empty()) {
		return Eigen::Vector2d(0.0, 0.0);
	}
	auto itr_x = std::max_element(polygon_.begin(), polygon_.end(),
		[](Eigen::Vector2d a, Eigen::Vector2d b) { return a(0) < b(0); });
	auto itr_y = std::max_element(polygon_.begin(), polygon_.end(),
		[](Eigen::Vector2d a, Eigen::Vector2d b) { return a(1) < b(1); });
	return Eigen::Vector2d((*itr_x)(0), (*itr_y)(1));
}

void SelectionPolygon::FillPolygon(int width, int height)
{
	// Standard scan conversion code. See reference:
	// http://alienryderflex.com/polygon_fill/
	if (IsEmpty()) return;
	is_closed_ = true;
	polygon_interior_mask_.PrepareImage(width, height, 1, 1);
	std::fill(polygon_interior_mask_.data_.begin(),
			polygon_interior_mask_.data_.end(), 0);
	std::vector<int> nodes;
	for (int y = 0; y < height; y++) {
		nodes.clear();
		for (size_t i = 0; i < polygon_.size(); i++) {
			size_t j = (i + 1) % polygon_.size();
			if ((polygon_[i](1) < y && polygon_[j](1) >= y) ||
					(polygon_[j](1) < y && polygon_[i](1) >= y)) {
				nodes.push_back((int)(polygon_[i](0) + (y - polygon_[i](1)) /
						(polygon_[j](1) - polygon_[i](1)) * (polygon_[j](0) -
						polygon_[i](0)) + 0.5));
			}
		}
		std::sort(nodes.begin(), nodes.end());
		for (size_t i = 0; i < nodes.size(); i+= 2) {
			if (nodes[i] >= width) {
				break;
			}
			if (nodes[i + 1] > 0) {
				if (nodes[i] < 0) nodes[i] = 0;
				if (nodes[i + 1] > width) nodes[i + 1] = width;
				for (int x = nodes[i]; x < nodes[i + 1]; x++) {
					polygon_interior_mask_.data_[x + y * width] = 1;
				}
			}
		}
	}
}

void SelectionPolygon::CropGeometry(const Geometry &input,
		const ViewControl &view, Geometry &output)
{
	if (IsEmpty() || polygon_type_ == POLYGON_UNFILLED) return;
	if (input.GetGeometryType() != output.GetGeometryType()) return;
	if (input.GetGeometryType() == Geometry::GEOMETRY_POINTCLOUD) {
		const auto &input_pointcloud = (const PointCloud &)input;
		auto &output_pointcloud = (PointCloud &)output;
		if (polygon_type_ == POLYGON_RECTANGLE) {
			CropPointCloudInRectangle(input_pointcloud, view,
					output_pointcloud);
		} else if (polygon_type_ == POLYGON_POLYGON) {
			CropPointCloudInPolygon(input_pointcloud, view, output_pointcloud);
		}
		PrintDebug("[CropGeometry] PointCloud size reduced from %d to %d.\n",
				(int)input_pointcloud.points_.size(),
				(int)output_pointcloud.points_.size());
	}
}

void SelectionPolygon::CropPointCloudInRectangle(const PointCloud &input,
		const ViewControl &view, PointCloud &output)
{
	output.Clear();
	bool has_normals = input.HasNormals();
	bool has_colors = input.HasColors();
	Eigen::Matrix4d mvp_matrix = view.GetMVPMatrix().cast<double>();
	double half_width = (double)view.GetWindowWidth() * 0.5;
	double half_height = (double)view.GetWindowHeight() * 0.5;
	auto min_bound = GetMinBound();
	auto max_bound = GetMaxBound();
	for (size_t i = 0; i < input.points_.size(); i++) {
		const auto &point = input.points_[i];
		Eigen::Vector4d pos = mvp_matrix * Eigen::Vector4d(point(0), point(1),
				point(2), 1.0);
		if (pos(3) == 0.0) break;
		pos /= pos(3);
		double x = (pos(0) + 1.0) * half_width;
		double y = (pos(1) + 1.0) * half_height;
		if (x >= min_bound(0) && x <= max_bound(0) &&
				y >= min_bound(1) && y <= max_bound(1)) {
			output.points_.push_back(point);
			if (has_normals) output.normals_.push_back(input.normals_[i]);
			if (has_colors) output.colors_.push_back(input.colors_[i]);
		}
	}
}

void SelectionPolygon::CropPointCloudInPolygon(const PointCloud &input,
		const ViewControl &view, PointCloud &output)
{
	output.Clear();
	bool has_normals = input.HasNormals();
	bool has_colors = input.HasColors();
	Eigen::Matrix4d mvp_matrix = view.GetMVPMatrix().cast<double>();
	double half_width = (double)view.GetWindowWidth() * 0.5;
	double half_height = (double)view.GetWindowHeight() * 0.5;
	std::vector<int> nodes;
	for (size_t i = 0; i < input.points_.size(); i++) {
		const auto &point = input.points_[i];
		Eigen::Vector4d pos = mvp_matrix * Eigen::Vector4d(point(0), point(1),
				point(2), 1.0);
		if (pos(3) == 0.0) break;
		pos /= pos(3);
		double x = (pos(0) + 1.0) * half_width;
		double y = (pos(1) + 1.0) * half_height;
		nodes.clear();
		for (size_t i = 0; i < polygon_.size(); i++) {
			size_t j = (i + 1) % polygon_.size();
			if ((polygon_[i](1) < y && polygon_[j](1) >= y) ||
					(polygon_[j](1) < y && polygon_[i](1) >= y)) {
				nodes.push_back((int)(polygon_[i](0) + (y - polygon_[i](1)) /
						(polygon_[j](1) - polygon_[i](1)) * (polygon_[j](0) -
						polygon_[i](0)) + 0.5));
			}
		}
		std::sort(nodes.begin(), nodes.end());
		auto loc = std::lower_bound(nodes.begin(), nodes.end(), x);
		if (std::distance(nodes.begin(), loc) % 2 == 1) {
			output.points_.push_back(point);
			if (has_normals) output.normals_.push_back(input.normals_[i]);
			if (has_colors) output.colors_.push_back(input.colors_[i]);
		}
	}
}

}	// namespace three
