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

#include "Open3D/Visualization/Utility/SelectionPolygon.h"

#include "Open3D/Geometry/PointCloud.h"
#include "Open3D/Geometry/TriangleMesh.h"
#include "Open3D/Utility/Console.h"
#include "Open3D/Visualization/Utility/GLHelper.h"
#include "Open3D/Visualization/Utility/SelectionPolygonVolume.h"
#include "Open3D/Visualization/Visualizer/ViewControl.h"
#include "Open3D/Visualization/Visualizer/ViewControlWithEditing.h"

namespace open3d {
namespace visualization {

SelectionPolygon &SelectionPolygon::Clear() {
    polygon_.clear();
    is_closed_ = false;
    polygon_interior_mask_.Clear();
    polygon_type_ = SectionPolygonType::Unfilled;
    return *this;
}

bool SelectionPolygon::IsEmpty() const {
    // A valid polygon, either close or open, should have at least 2 vertices.
    return polygon_.size() <= 1;
}

Eigen::Vector2d SelectionPolygon::GetMinBound() const {
    if (polygon_.empty()) {
        return Eigen::Vector2d(0.0, 0.0);
    }
    auto itr_x = std::min_element(
            polygon_.begin(), polygon_.end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
                return a(0) < b(0);
            });
    auto itr_y = std::min_element(
            polygon_.begin(), polygon_.end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
                return a(1) < b(1);
            });
    return Eigen::Vector2d((*itr_x)(0), (*itr_y)(1));
}

Eigen::Vector2d SelectionPolygon::GetMaxBound() const {
    if (polygon_.empty()) {
        return Eigen::Vector2d(0.0, 0.0);
    }
    auto itr_x = std::max_element(
            polygon_.begin(), polygon_.end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
                return a(0) < b(0);
            });
    auto itr_y = std::max_element(
            polygon_.begin(), polygon_.end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
                return a(1) < b(1);
            });
    return Eigen::Vector2d((*itr_x)(0), (*itr_y)(1));
}

void SelectionPolygon::FillPolygon(int width, int height) {
    // Standard scan conversion code. See reference:
    // http://alienryderflex.com/polygon_fill/
    if (IsEmpty()) return;
    is_closed_ = true;
    polygon_interior_mask_.Prepare(width, height, 1, 1);
    std::fill(polygon_interior_mask_.data_.begin(),
              polygon_interior_mask_.data_.end(), 0);
    std::vector<int> nodes;
    for (int y = 0; y < height; y++) {
        nodes.clear();
        for (size_t i = 0; i < polygon_.size(); i++) {
            size_t j = (i + 1) % polygon_.size();
            if ((polygon_[i](1) < y && polygon_[j](1) >= y) ||
                (polygon_[j](1) < y && polygon_[i](1) >= y)) {
                nodes.push_back(
                        (int)(polygon_[i](0) +
                              (y - polygon_[i](1)) /
                                      (polygon_[j](1) - polygon_[i](1)) *
                                      (polygon_[j](0) - polygon_[i](0)) +
                              0.5));
            }
        }
        std::sort(nodes.begin(), nodes.end());
        for (size_t i = 0; i < nodes.size(); i += 2) {
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

std::shared_ptr<geometry::PointCloud> SelectionPolygon::CropPointCloud(
        const geometry::PointCloud &input, const ViewControl &view) {
    if (IsEmpty()) {
        return std::make_shared<geometry::PointCloud>();
    }
    switch (polygon_type_) {
        case SectionPolygonType::Rectangle:
            return CropPointCloudInRectangle(input, view);
        case SectionPolygonType::Polygon:
            return CropPointCloudInPolygon(input, view);
        case SectionPolygonType::Unfilled:
        default:
            return std::shared_ptr<geometry::PointCloud>();
    }
}

std::shared_ptr<geometry::TriangleMesh> SelectionPolygon::CropTriangleMesh(
        const geometry::TriangleMesh &input, const ViewControl &view) {
    if (IsEmpty()) {
        return std::make_shared<geometry::TriangleMesh>();
    }
    if (input.HasVertices() && !input.HasTriangles()) {
        utility::LogWarning(
                "geometry::TriangleMesh contains vertices, but no triangles; "
                "cropping will always yield an empty "
                "geometry::TriangleMesh.\n");
        return std::make_shared<geometry::TriangleMesh>();
    }
    switch (polygon_type_) {
        case SectionPolygonType::Rectangle:
            return CropTriangleMeshInRectangle(input, view);
        case SectionPolygonType::Polygon:
            return CropTriangleMeshInPolygon(input, view);
        case SectionPolygonType::Unfilled:
        default:
            return std::shared_ptr<geometry::TriangleMesh>();
    }
}

std::shared_ptr<SelectionPolygonVolume>
SelectionPolygon::CreateSelectionPolygonVolume(const ViewControl &view) {
    auto volume = std::make_shared<SelectionPolygonVolume>();
    const auto &editing_view = (const ViewControlWithEditing &)view;
    if (editing_view.IsLocked() == false ||
        editing_view.GetEditingMode() == ViewControlWithEditing::FreeMode) {
        return volume;
    }
    int idx = 0;
    switch (editing_view.GetEditingMode()) {
        case ViewControlWithEditing::OrthoNegativeX:
        case ViewControlWithEditing::OrthoPositiveX:
            volume->orthogonal_axis_ = "X";
            idx = 0;
            break;
        case ViewControlWithEditing::OrthoNegativeY:
        case ViewControlWithEditing::OrthoPositiveY:
            volume->orthogonal_axis_ = "Y";
            idx = 1;
            break;
        case ViewControlWithEditing::OrthoNegativeZ:
        case ViewControlWithEditing::OrthoPositiveZ:
            volume->orthogonal_axis_ = "Z";
            idx = 2;
            break;
        default:
            break;
    }
    for (const auto &point : polygon_) {
        auto point3d = GLHelper::Unproject(
                Eigen::Vector3d(point(0), point(1), 1.0), view.GetMVPMatrix(),
                view.GetWindowWidth(), view.GetWindowHeight());
        point3d(idx) = 0.0;
        volume->bounding_polygon_.push_back(point3d);
    }
    const auto &boundingbox = view.GetBoundingBox();
    double axis_len = boundingbox.max_bound_(idx) - boundingbox.min_bound_(idx);
    volume->axis_min_ = boundingbox.min_bound_(idx) - axis_len;
    volume->axis_max_ = boundingbox.max_bound_(idx) + axis_len;
    return volume;
}

std::shared_ptr<geometry::PointCloud>
SelectionPolygon::CropPointCloudInRectangle(const geometry::PointCloud &input,
                                            const ViewControl &view) {
    return input.SelectDownSample(CropInRectangle(input.points_, view));
}

std::shared_ptr<geometry::PointCloud> SelectionPolygon::CropPointCloudInPolygon(
        const geometry::PointCloud &input, const ViewControl &view) {
    return input.SelectDownSample(CropInPolygon(input.points_, view));
}

std::shared_ptr<geometry::TriangleMesh>
SelectionPolygon::CropTriangleMeshInRectangle(
        const geometry::TriangleMesh &input, const ViewControl &view) {
    return input.SelectDownSample(CropInRectangle(input.vertices_, view));
}

std::shared_ptr<geometry::TriangleMesh>
SelectionPolygon::CropTriangleMeshInPolygon(const geometry::TriangleMesh &input,
                                            const ViewControl &view) {
    return input.SelectDownSample(CropInPolygon(input.vertices_, view));
}

std::vector<size_t> SelectionPolygon::CropInRectangle(
        const std::vector<Eigen::Vector3d> &input, const ViewControl &view) {
    std::vector<size_t> output_index;
    Eigen::Matrix4d mvp_matrix = view.GetMVPMatrix().cast<double>();
    double half_width = (double)view.GetWindowWidth() * 0.5;
    double half_height = (double)view.GetWindowHeight() * 0.5;
    auto min_bound = GetMinBound();
    auto max_bound = GetMaxBound();
    utility::ConsoleProgressBar progress_bar((int64_t)input.size(),
                                             "Cropping geometry: ");
    for (size_t i = 0; i < input.size(); i++) {
        ++progress_bar;
        const auto &point = input[i];
        Eigen::Vector4d pos =
                mvp_matrix * Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        if (pos(3) == 0.0) break;
        pos /= pos(3);
        double x = (pos(0) + 1.0) * half_width;
        double y = (pos(1) + 1.0) * half_height;
        if (x >= min_bound(0) && x <= max_bound(0) && y >= min_bound(1) &&
            y <= max_bound(1)) {
            output_index.push_back(i);
        }
    }
    return output_index;
}

std::vector<size_t> SelectionPolygon::CropInPolygon(
        const std::vector<Eigen::Vector3d> &input, const ViewControl &view) {
    std::vector<size_t> output_index;
    Eigen::Matrix4d mvp_matrix = view.GetMVPMatrix().cast<double>();
    double half_width = (double)view.GetWindowWidth() * 0.5;
    double half_height = (double)view.GetWindowHeight() * 0.5;
    std::vector<double> nodes;
    utility::ConsoleProgressBar progress_bar((int64_t)input.size(),
                                             "Cropping geometry: ");
    for (size_t k = 0; k < input.size(); k++) {
        ++progress_bar;
        const auto &point = input[k];
        Eigen::Vector4d pos =
                mvp_matrix * Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        if (pos(3) == 0.0) break;
        pos /= pos(3);
        double x = (pos(0) + 1.0) * half_width;
        double y = (pos(1) + 1.0) * half_height;
        nodes.clear();
        for (size_t i = 0; i < polygon_.size(); i++) {
            size_t j = (i + 1) % polygon_.size();
            if ((polygon_[i](1) < y && polygon_[j](1) >= y) ||
                (polygon_[j](1) < y && polygon_[i](1) >= y)) {
                nodes.push_back(polygon_[i](0) +
                                (y - polygon_[i](1)) /
                                        (polygon_[j](1) - polygon_[i](1)) *
                                        (polygon_[j](0) - polygon_[i](0)));
            }
        }
        std::sort(nodes.begin(), nodes.end());
        auto loc = std::lower_bound(nodes.begin(), nodes.end(), x);
        if (std::distance(nodes.begin(), loc) % 2 == 1) {
            output_index.push_back(k);
        }
    }
    return output_index;
}

}  // namespace visualization
}  // namespace open3d
