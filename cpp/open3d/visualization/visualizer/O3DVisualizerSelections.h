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

#include <Eigen/Core>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "open3d/visualization/gui/SceneWidget.h"

namespace open3d {

namespace geometry {
class Geometry3D;
}  // namespace geometry

namespace t {
namespace geometry {
class Geometry;
}  // namespace geometry
}  // namespace t

namespace visualization {
namespace visualizer {

/// Internal class that acts as a selections model + controller for
/// O3DVisualizer
class O3DVisualizerSelections {
public:
    struct SelectedIndex {
        size_t index;  /// the index of the point within the object
        size_t order;  /// A monotonically increasing number indicating the
                       /// relative order of when the index was selected
        Eigen::Vector3d point;  /// the point in R^3 (for convenience)

        bool operator!=(const SelectedIndex& rhs) const {
            return index != rhs.index;
        }
        bool operator<(const SelectedIndex& rhs) const {
            return index < rhs.index;
        }
    };
    using SelectionSet = std::map<std::string,  // name of object
                                  std::set<SelectedIndex>>;

public:
    O3DVisualizerSelections(gui::SceneWidget& widget3d);
    ~O3DVisualizerSelections();

    void NewSet();
    void RemoveSet(int index);
    void SelectSet(int index);
    size_t GetNumberOfSets() const;

    void SelectIndices(
            const std::map<std::string,
                           std::vector<std::pair<size_t, Eigen::Vector3d>>>&
                    indices);
    void UnselectIndices(
            const std::map<std::string,
                           std::vector<std::pair<size_t, Eigen::Vector3d>>>&
                    indices);
    std::vector<SelectionSet> GetSets();

    // Since the points are spheres, the radius is in world coordinates
    void SetPointSize(double radius_world);

    void MakeActive();
    void MakeInactive();
    bool IsActive() const;

    void SetSelectableGeometry(
            const std::vector<gui::SceneWidget::PickableGeometry>& geometry);

private:
    void UpdatePointSize();
    void UpdateSelectionGeometry();

private:
    gui::SceneWidget& widget3d_;
    int next_id_ = 1;

    struct SelectedPoints {
        std::string name;
        SelectionSet indices;
    };

    double point_size_ = 3.0;
    bool is_active_ = false;
    size_t pick_order_ = 0;
    std::vector<SelectedPoints> sets_;
    int current_set_index_ = -1;

    bool point_size_changed_ = false;
};

}  // namespace visualizer
}  // namespace visualization
}  // namespace open3d
