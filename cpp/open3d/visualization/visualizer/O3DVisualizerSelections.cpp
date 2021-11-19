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

#include "open3d/visualization/visualizer/O3DVisualizerSelections.h"

#include <assert.h>

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/visualization/gui/SceneWidget.h"
#include "open3d/visualization/rendering/MaterialRecord.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/Scene.h"

namespace open3d {
namespace visualization {
namespace visualizer {

namespace {
rendering::MaterialRecord MakeMaterial() {
    rendering::MaterialRecord m;
    m.shader = "defaultUnlit";
    m.base_color = {1.0f, 0.0f, 1.0f, 1.0f};
    return m;
}
}  // namespace

// ----------------------------------------------------------------------------
O3DVisualizerSelections::O3DVisualizerSelections(gui::SceneWidget &widget3d)
    : widget3d_(widget3d) {}

O3DVisualizerSelections::~O3DVisualizerSelections() {}

void O3DVisualizerSelections::NewSet() {
    auto name = std::string("__selection_") + std::to_string(next_id_++);
    sets_.push_back({name, {}});
    if (current_set_index_ < 0) {
        current_set_index_ = int(sets_.size()) - 1;
    }
}

void O3DVisualizerSelections::RemoveSet(int index) {
    auto scene = widget3d_.GetScene();
    if (scene->HasGeometry(sets_[index].name)) {
        scene->RemoveGeometry(sets_[index].name);
    }
    sets_.erase(sets_.begin() + index);
    current_set_index_ = std::min(int(sets_.size()) - 1, current_set_index_);

    if (!sets_.empty() && scene->HasGeometry(sets_[current_set_index_].name)) {
        scene->ShowGeometry(sets_[current_set_index_].name, true);
    }
}

void O3DVisualizerSelections::SelectSet(int index) {
    auto scene = widget3d_.GetScene();
    if (scene->HasGeometry(sets_[current_set_index_].name)) {
        scene->ShowGeometry(sets_[current_set_index_].name, false);
    }

    current_set_index_ = index;

    if (!sets_.empty() && scene->HasGeometry(sets_[current_set_index_].name)) {
        scene->ShowGeometry(sets_[current_set_index_].name, true);
    }
}

size_t O3DVisualizerSelections::GetNumberOfSets() const { return sets_.size(); }

void O3DVisualizerSelections::SelectIndices(
        const std::map<std::string,
                       std::vector<std::pair<size_t, Eigen::Vector3d>>>
                &indices) {
    if (!sets_.empty()) {
        auto &selection = sets_[current_set_index_];
        for (auto &name_indices : indices) {
            auto &name = name_indices.first;
            for (auto idx_pt : name_indices.second) {
                auto &idx = idx_pt.first;
                auto &p = idx_pt.second;
                selection.indices[name].insert({idx, pick_order_, p});
            }
        }
        pick_order_ += 1;

        UpdateSelectionGeometry();
    }
}

void O3DVisualizerSelections::UnselectIndices(
        const std::map<std::string,
                       std::vector<std::pair<size_t, Eigen::Vector3d>>>
                &indices) {
    if (!sets_.empty()) {
        auto &selection = sets_[current_set_index_];
        for (auto &name_indices : indices) {
            auto &name = name_indices.first;
            for (auto idx_pt : name_indices.second) {
                auto &idx = idx_pt.first;
                auto &p = idx_pt.second;
                selection.indices[name].erase({idx, pick_order_, p});
            }
        }
        pick_order_ += 1;

        UpdateSelectionGeometry();
    }
}

void O3DVisualizerSelections::UpdateSelectionGeometry() {
    auto scene = widget3d_.GetScene();
    auto &selection = sets_[current_set_index_];
    if (scene->HasGeometry(selection.name)) {
        scene->RemoveGeometry(selection.name);
    }
    std::vector<Eigen::Vector3d> points;
    points.reserve(selection.indices.size());
    for (auto &kv : selection.indices) {
        for (auto &i : kv.second) {
            points.push_back(i.point);
        }
    }

    if (!points.empty()) {
        geometry::TriangleMesh spheres;
        for (auto &p : points) {
            auto ps = geometry::TriangleMesh::CreateSphere(point_size_, 10);
            ps->Translate(p);
            spheres += *ps;
        }
        scene->AddGeometry(selection.name, &spheres, MakeMaterial());
        scene->GetScene()->GeometryShadows(selection.name, false, false);
    }

    widget3d_.ForceRedraw();
}

std::vector<O3DVisualizerSelections::SelectionSet>
O3DVisualizerSelections::GetSets() {
    std::vector<SelectionSet> all;
    all.reserve(sets_.size());
    for (auto &s : sets_) {
        all.push_back(s.indices);
    }
    return all;
}

void O3DVisualizerSelections::SetPointSize(double radius_world) {
    point_size_ = radius_world;
    if (IsActive()) {
        UpdatePointSize();
    } else {
        point_size_changed_ = true;
    }
}

void O3DVisualizerSelections::MakeActive() {
    if (is_active_) {
        utility::LogError("Already active.");
    }

    is_active_ = true;
    widget3d_.SetViewControls(gui::SceneWidget::Controls::PICK_POINTS);
    auto scene = widget3d_.GetScene();

    if (point_size_changed_) {
        UpdatePointSize();
        point_size_changed_ = false;
    }

    auto &selection = sets_[current_set_index_];
    if (scene->HasGeometry(selection.name)) {
        scene->ShowGeometry(selection.name, true);
    }
}

bool O3DVisualizerSelections::IsActive() const { return is_active_; }

void O3DVisualizerSelections::MakeInactive() {
    auto scene = widget3d_.GetScene();

    auto &selection = sets_[current_set_index_];
    if (scene->HasGeometry(selection.name)) {
        scene->ShowGeometry(selection.name, false);
    }

    is_active_ = false;
}

void O3DVisualizerSelections::SetSelectableGeometry(
        const std::vector<gui::SceneWidget::PickableGeometry> &geometry) {
    widget3d_.SetPickableGeometry(geometry);
}

void O3DVisualizerSelections::UpdatePointSize() { UpdateSelectionGeometry(); }

}  // namespace visualizer
}  // namespace visualization
}  // namespace open3d
