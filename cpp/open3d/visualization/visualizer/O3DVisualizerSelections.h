// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
