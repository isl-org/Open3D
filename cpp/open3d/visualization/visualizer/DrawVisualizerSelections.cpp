// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/visualization/visualizer/DrawVisualizerSelections.h"

#include <assert.h>

#include <sstream>

#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/visualization/gui/SceneWidget.h"
#include "open3d/visualization/rendering/Material.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/Scene.h"

namespace open3d {
namespace visualization {
namespace visualizer {

namespace {
rendering::Material MakeMaterial(int point_size) {
    rendering::Material m;
    m.shader = "unlitPolygonOffset";
    m.point_size = point_size;
    //    m.depth_offset = -0.000010f;
    return m;
}
}  // namespace

class SelectionIndexLookup {
private:
    struct Obj {
        std::string name;
        size_t start_index;

        Obj(const std::string &n, size_t start) : name(n), start_index(start){};
    };

public:
    void Clear() { objects_.clear(); }

    // start_index must be larger than all previously added items
    void Add(const std::string &name, size_t start_index) {
        assert(objects_.empty() || objects_.back().start_index < start_index);
        objects_.emplace_back(name, start_index);
        assert(objects_[0].start_index == 0);
    }

    const Obj &ObjectForIndex(size_t index) {
        if (objects_.size() == 1) {
            return objects_[0];
        } else {
            auto next = std::upper_bound(objects_.begin(), objects_.end(),
                                         index, [](size_t value, const Obj &o) {
                                             return value < o.start_index;
                                         });
            assert(next != objects_.end());  // first object != 0
            if (next == objects_.end()) {
                return objects_.back();

            } else {
                --next;
                return *next;
            }
        }
    }

private:
    std::vector<Obj> objects_;
};

// ----------------------------------------------------------------------------
DrawVisualizerSelections::DrawVisualizerSelections(gui::SceneWidget &widget3d)
    : widget3d_(widget3d) {
    current_.lookup = new SelectionIndexLookup();
}

DrawVisualizerSelections::~DrawVisualizerSelections() {
    delete current_.lookup;
}

void DrawVisualizerSelections::NewSet() {
    std::stringstream s;
    s << "__selection_" << next_id_++;
    sets_.push_back({s.str()});
    if (current_.index < 0) {
        current_.index = int(sets_.size()) - 1;
    }
}

void DrawVisualizerSelections::RemoveSet(int index) {
    auto scene = widget3d_.GetScene();
    if (scene->HasGeometry(sets_[index].name)) {
        scene->RemoveGeometry(sets_[index].name);
    }
    sets_.erase(sets_.begin() + index);
    current_.index = std::min(int(sets_.size()) - 1, current_.index);

    if (scene->HasGeometry(sets_[current_.index].name)) {
        scene->ShowGeometry(sets_[current_.index].name, true);
    }
}

void DrawVisualizerSelections::SelectSet(int index) {
    auto scene = widget3d_.GetScene();
    if (scene->HasGeometry(sets_[current_.index].name)) {
        scene->ShowGeometry(sets_[current_.index].name, false);
    }

    current_.index = index;

    if (scene->HasGeometry(sets_[current_.index].name)) {
        scene->ShowGeometry(sets_[current_.index].name, true);
    }
}

size_t DrawVisualizerSelections::GetNumberOfSets() const {
    return sets_.size();
}

void DrawVisualizerSelections::SelectIndices(
        const std::vector<size_t> &indices) {
    auto &selection = sets_[current_.index];
    for (auto idx : indices) {
        auto &o = current_.lookup->ObjectForIndex(idx);
        auto p = current_.selectable_points[idx];
        selection.indices[o.name].insert({idx - o.start_index, pick_order_, p});
    }
    pick_order_ += 1;

    UpdateSelectionGeometry();
}

void DrawVisualizerSelections::UnselectIndices(
        const std::vector<size_t> &indices) {
    auto &selection = sets_[current_.index];
    for (auto idx : indices) {
        auto &o = current_.lookup->ObjectForIndex(idx);
        auto p = current_.selectable_points[idx];
        selection.indices[o.name].erase({idx - o.start_index, pick_order_, p});
    }
    pick_order_ += 1;

    UpdateSelectionGeometry();
}

void DrawVisualizerSelections::UpdateSelectionGeometry() {
    auto scene = widget3d_.GetScene();
    auto &selection = sets_[current_.index];
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

    // Hack: Filament doesn't like a 1 point object, because it wants a
    // non-empty bounding box. So if we only have one point, offset it
    // ever-so-slightly to make Filament happy.
    if (points.size() == 1) {
        points.push_back({points[0].x() + 0.000001, points[0].y() + 0.000001,
                          points[0].z() + 0.000001});
    }
    if (!points.empty()) {
        auto cloud = std::make_shared<geometry::PointCloud>(points);
        cloud->colors_.reserve(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            cloud->colors_.push_back({1.0, 0.0, 1.0});
        }
        scene->AddGeometry(selection.name, cloud, MakeMaterial(point_size_));
        scene->GetScene()->GeometryShadows(selection.name, false, false);
    }
    widget3d_.ForceRedraw();
}

std::vector<DrawVisualizerSelections::SelectionSet>
DrawVisualizerSelections::GetSets() {
    std::vector<SelectionSet> all;
    all.reserve(sets_.size());
    for (auto &s : sets_) {
        all.push_back(s.indices);
    }
    return all;
}

void DrawVisualizerSelections::SetPointSize(int px) {
    point_size_ = px;
    if (IsActive()) {
        UpdatePointSize();
    } else {
        point_size_changed_ = true;
    }
}

void DrawVisualizerSelections::MakeActive() {
    assert(!is_active_);

    is_active_ = true;
    widget3d_.SetViewControls(gui::SceneWidget::Controls::PICK_POINTS);
    auto scene = widget3d_.GetScene();

    if (point_size_changed_) {
        UpdatePointSize();
        point_size_changed_ = false;
    }

    auto &selection = sets_[current_.index];
    if (scene->HasGeometry(selection.name)) {
        scene->ShowGeometry(selection.name, true);
    }
}

bool DrawVisualizerSelections::IsActive() const { return is_active_; }

void DrawVisualizerSelections::MakeInactive() {
    auto scene = widget3d_.GetScene();

    auto &selection = sets_[current_.index];
    if (scene->HasGeometry(selection.name)) {
        scene->ShowGeometry(selection.name, false);
    }

    is_active_ = false;
}

void DrawVisualizerSelections::StartSelectablePoints() {
    current_.selectable_points.clear();
    current_.lookup->Clear();
}

void DrawVisualizerSelections::AddSelectablePoints(
        const std::string &name,
        geometry::Geometry3D *geom,
        t::geometry::Geometry *tgeom) {
    auto &points = current_.selectable_points;
    current_.lookup->Add(name, points.size());

    auto cloud = dynamic_cast<geometry::PointCloud *>(geom);
    auto tcloud = dynamic_cast<t::geometry::PointCloud *>(tgeom);
    auto mesh = dynamic_cast<geometry::TriangleMesh *>(geom);
    auto tmesh = dynamic_cast<t::geometry::TriangleMesh *>(tgeom);
    if (cloud) {
        points.insert(points.end(), cloud->points_.begin(),
                      cloud->points_.end());
    } else if (mesh) {
        points.insert(points.end(), mesh->vertices_.begin(),
                      mesh->vertices_.end());
    } else if (tcloud || tmesh) {
        const auto &tpoints =
                (tcloud ? tcloud->GetPoints() : tmesh->GetVertices());
        const size_t n = tpoints.GetSize();
        float *pts = (float *)tpoints.AsTensor().GetDataPtr();
        points.reserve(points.size() + n);
        for (size_t i = 0; i < n; i += 3) {
            points.emplace_back(double(pts[i]), double(pts[i + 1]),
                                double(pts[i + 2]));
        }
    }
}

void DrawVisualizerSelections::EndSelectablePoints() {
    widget3d_.SetPickablePoints(current_.selectable_points);
}

void DrawVisualizerSelections::UpdatePointSize() {
    auto scene = widget3d_.GetScene();
    auto material = MakeMaterial(point_size_);
    for (auto &s : sets_) {
        if (scene->HasGeometry(s.name)) {
            scene->GetScene()->OverrideMaterial(s.name, material);
        }
    }
}

}  // namespace visualizer
}  // namespace visualization
}  // namespace open3d
