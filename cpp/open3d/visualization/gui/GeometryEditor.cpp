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

#include "open3d/visualization/gui/GeometryEditor.h"

#include "open3d/geometry/Geometry.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/gui/Color.h"
#include "open3d/visualization/gui/Events.h"
#include "open3d/visualization/gui/Util.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/View.h"

namespace open3d {
namespace visualization {
namespace gui {

static auto col_start = colorToImguiRGBA({0.5f, 0.0f, 0.5f, 1.0f});
static auto col_end = colorToImguiRGBA({1.0f, 0.3f, 0.8f, 1.0f});
static auto col_fill = colorToImguiRGBA({0.5f, 0.0f, 0.5f, 0.5f});
static auto col_line = colorToImguiRGBA({1.0f, 0.0f, 0.0f, 1.0f});

template <class T>
class Seg {
    class Point {
    public:
        Point(int x_, int y_) {
            x = x_;
            y = y_;
        }
        T x, y;
    };

public:
    Seg(const Eigen::Vector2<T> &p0, const Eigen::Vector2<T> &p1)
        : p0_(p0), p1_(p1) {}
    const Eigen::Vector2<T> &p0_, &p1_;
    bool cross(const Seg &other) const {
        Point p1(p0_.x(), p0_.y());
        Point p2(p1_.x(), p1_.y());
        Point q1(other.p0_.x(), other.p0_.y());
        Point q2(other.p1_.x(), other.p1_.y());
        return doIntersect(p1, p2, q1, q2);
    }

private:
    // Given three collinear points p, q, r, the function checks if
    // point q lies on line segment 'pr'
    bool onSegment(const Point &p, const Point &q, const Point &r) const {
        if (q.x <= std::max(p.x, r.x) && q.x >= std::min(p.x, r.x) &&
            q.y <= std::max(p.y, r.y) && q.y >= std::min(p.y, r.y))
            return true;

        return false;
    }

    // To find orientation of ordered triplet (p, q, r).
    // The function returns following values
    // 0 --> p, q and r are collinear
    // 1 --> Clockwise
    // 2 --> Counterclockwise
    int orientation(const Point &p, const Point &q, const Point &r) const {
        // See https://www.geeksforgeeks.org/orientation-3-ordered-points/
        // for details of below formula.
        int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
        if (val == 0) return 0;    // collinear
        return (val > 0) ? 1 : 2;  // clock or counterclock wise
    }

    // The main function that returns true if line segment 'p1q1'
    // and 'p2q2' intersect.
    bool doIntersect(const Point &p1,
                     const Point &q1,
                     const Point &p2,
                     const Point &q2) const {
        // Find the four orientations needed for general and
        // special cases
        int o1 = orientation(p1, q1, p2);
        int o2 = orientation(p1, q1, q2);
        int o3 = orientation(p2, q2, p1);
        int o4 = orientation(p2, q2, q1);

        // General case
        if (o1 != o2 && o3 != o4) return true;

        // Special Cases
        // p1, q1 and p2 are collinear and p2 lies on segment p1q1
        if (o1 == 0 && onSegment(p1, p2, q1)) return true;
        // p1, q1 and q2 are collinear and q2 lies on segment p1q1
        if (o2 == 0 && onSegment(p1, q2, q1)) return true;
        // p2, q2 and p1 are collinear and p1 lies on segment p2q2
        if (o3 == 0 && onSegment(p2, p1, q2)) return true;
        // p2, q2 and q1 are collinear and q1 lies on segment p2q2
        if (o4 == 0 && onSegment(p2, q1, q2)) return true;
        return false;  // Doesn't fall in any of the above cases
    }
};
GeometryEditor::GeometryEditor(rendering::Open3DScene *scene) {
    scene_ = scene;
    camera_ = scene->GetCamera();
}
const std::vector<Eigen::Vector3d> &GeometryEditor::GetPoints() {
    static std::vector<Eigen::Vector3d> empty;

    auto gtype = target_->GetGeometryType();
    if (gtype == geometry::Geometry::GeometryType::TriangleMesh) {
        return ((const geometry::TriangleMesh &)*target_).vertices_;
    }
    if (gtype == geometry::Geometry::GeometryType::PointCloud) {
        return ((const geometry::PointCloud &)*target_).points_;
    }
    return empty;
}

bool GeometryEditor::Start(GeometryEditor::Target target,
                           std::function<void(bool)> selectionCallback) {
    if (!target) {
        return false;
    }

    Stop();

    auto gtype = target->GetGeometryType();
    if (gtype == geometry::Geometry::GeometryType::TriangleMesh ||
        gtype == geometry::Geometry::GeometryType::PointCloud) {
        target_ = target;
    } else {
        utility::LogWarning(
                "Geometry editor accept point cloud and"
                "triangle mesh only");
        return false;
    }

    selection_callback_ = selectionCallback;
    editable_ = false;
    CheckEditable();  // in case called in editing
    return true;
}
void GeometryEditor::Stop() {
    if (Started()) {
        SetSelection(SelectionType::None);
        CheckEditable();
        selection_callback_ = nullptr;
        target_.reset();
    }
}

bool GeometryEditor::Started() { return bool(target_); }

std::vector<size_t> GeometryEditor::CollectSelectedIndices() {
    if (!AllowEdit()) {
        return std::vector<size_t>{};
    }

    switch (type_) {
        case SelectionType::Polygon:
            return CropPolygon();
        case SelectionType::Rectangle:
            return CropRectangle();
        case SelectionType::Circle:
            return CropCircle();
        default:
            return std::vector<size_t>{};
    }
}
bool GeometryEditor::SetSelection(GeometryEditor::SelectionType type) {
    if (type_ != type) {
        type_ = type;
        selection_.clear();
        CheckEditable();
        return true;
    }
    return false;
}
bool GeometryEditor::AllowEdit() {
    bool editable = false;
    if (Started()) {
        if (type_ == SelectionType::Rectangle ||
            type_ == SelectionType::Circle) {
            editable = selection_.size() > 1;
        } else if (type_ == SelectionType::Polygon) {
            editable = selection_.size() > 2;
        }
    }
    return editable;
}
void GeometryEditor::CheckEditable() {
    auto state = AllowEdit();
    if (editable_ != state && selection_callback_) {
        editable_ = state;
        selection_callback_(editable_);
    }
}
Widget::EventResult GeometryEditor::Mouse(const MouseEvent &e) {
    if (!Started()) {
        return Widget::EventResult::DISCARD;
    }
    if (e.type == MouseEvent::Type::BUTTON_DOWN) {
        if (e.button.button == MouseButton::MIDDLE &&
            type_ != SelectionType::None) {
            SetSelection(SelectionType::None);
            return Widget::EventResult::CONSUMED;
        }

        if (e.button.button == MouseButton::LEFT) {
            auto next = SelectionType::None;
            if (e.button.count == 1) {
                if (e.modifiers == int(KeyModifier::ALT)) {
                    next = SelectionType::Rectangle;
                } else if (e.modifiers == int(KeyModifier::CTRL)) {
                    next = SelectionType::Polygon;
                } else if (e.modifiers == int(KeyModifier::SHIFT)) {
                    next = SelectionType::Circle;
                }
            }
            SetSelection(next);
            if (type_ == SelectionType::None) {
                return Widget::EventResult::DISCARD;
            }
            if (type_ == SelectionType::Rectangle ||
                type_ == SelectionType::Circle) {
                selection_.clear();
            }
            AddPoint(e.x, e.y);
            return Widget::EventResult::CONSUMED;
        }

        if (e.button.button == MouseButton::RIGHT && e.button.count == 1 &&
            type_ != SelectionType::None) {
            if (type_ == SelectionType::Polygon) {
                if (!selection_.empty()) {
                    selection_.pop_back();
                    CheckEditable();
                }
            } else {
                SetSelection(SelectionType::None);
            }
            return Widget::EventResult::CONSUMED;
        }
    } else if (e.type == MouseEvent::Type::DRAG &&
               e.button.button == MouseButton::LEFT) {
        if (type_ == SelectionType::Rectangle ||
            type_ == SelectionType::Circle) {
            AddPoint(e.x, e.y);
            return Widget::EventResult::CONSUMED;
        }
        if (type_ == SelectionType::Polygon) {
            UpdatePolygonPoint(e.x, e.y);
            return Widget::EventResult::CONSUMED;
        }
    }
    return Widget::EventResult::DISCARD;
}
Widget::DrawResult GeometryEditor::Draw(const DrawContext &context,
                                        const Rect &frame) {
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    switch (type_) {
        case SelectionType::None:
            break;
        case SelectionType::Rectangle:
            if (selection_.size() == 2) {
                auto p0 = PointAt(0, frame.x, frame.y);
                auto p1 = PointAt(1, frame.x, frame.y);
                draw_list->AddRectFilled({p0.x(), p0.y()}, {p1.x(), p1.y()},
                                         col_fill);
            }
            break;
        case SelectionType::Polygon: {
            if (selection_.size() >= 2) {
                for (auto i = 0; i < (int)selection_.size(); i++) {
                    auto p0 = PointAt(i, frame.x, frame.y);
                    auto p1 = PointAt((i + 1) % (int(selection_.size())),
                                      frame.x, frame.y);
                    draw_list->AddLine({float(p0.x()), float(p0.y())},
                                       {float(p1.x()), float(p1.y())}, col_line,
                                       2);

                    auto p = PointAt(selection_.size() - 1, frame.x, frame.y);
                    draw_list->AddCircleFilled({p.x(), p.y()}, 10, col_end, 10);
                }
            }
            if (!selection_.empty()) {
                auto p = PointAt(0, frame.x, frame.y);
                draw_list->AddCircleFilled({p.x(), p.y()}, 10, col_start, 10);
            }
            break;
        }
        case SelectionType::Circle:
            if (selection_.size() == 2) {
                auto p0 = PointAt(0, frame.x, frame.y);
                auto p1 = PointAt(1, frame.x, frame.y);
                auto radius = (p0 - p1).norm();
                int segs = int(radius * 12 / 20);
                draw_list->AddCircleFilled({p0.x(), p0.y()}, radius, col_fill,
                                           segs);
            }
            break;
    }
    return Widget::DrawResult::NONE;
}
bool GeometryEditor::CheckPolygonPoint(int x, int y) {
    Eigen::Vector2i p(x, y);
    auto segs = selection_.size() - 1;
    auto discard = false;
    if (segs >= 2) {
        // seg by last point and new
        Seg<int> s0(selection_[selection_.size() - 1], p);
        // seg by first point and new
        Seg<int> s1(selection_[0], p);

        for (size_t idx = 0; !discard && idx < selection_.size() - 1; ++idx) {
            Seg<int> s(selection_[idx], selection_[idx + 1]);
            bool last = idx == selection_.size() - 2;
            bool first = idx == 0;
            discard = (!last && s0.cross(s)) || (!first && s1.cross(s));
        }
    }
    return !discard;
}
void GeometryEditor::UpdatePolygonPoint(int x, int y) {
    if (selection_.empty()) {
        return;
    }
    // update last selection with current, pop back first
    // and check if (x, y) is crossing to current selections.
    auto last = selection_.back();
    selection_.pop_back();
    if (!CheckPolygonPoint(x, y)) {
        selection_.push_back(last);
    } else {
        selection_.emplace_back(x, y);
    }
}
void GeometryEditor::AddPoint(int x, int y) {
    if (type_ == SelectionType::None) {
        return;
    }
    if (selection_.empty()) {
        selection_.emplace_back(x, y);
        return;
    }
    switch (type_) {
        case SelectionType::Rectangle:
        case SelectionType::Circle:
            if (selection_.size() == 1) {
                selection_.emplace_back(x, y);
            } else {
                selection_.back() = Eigen::Vector2i(x, y);
            }
            break;
        case SelectionType::Polygon:
            if (CheckPolygonPoint(x, y)) {
                selection_.emplace_back(x, y);
            }
            break;
        default:
            break;
    }
    CheckEditable();
}
Eigen::Vector2f GeometryEditor::PointAt(int i) { return PointAt(i, 0, 0); }
Eigen::Vector2f GeometryEditor::PointAt(int i, int x, int y) {
    auto &p = selection_[i];
    return Eigen::Vector2f{float(p.x() + x), float(p.y() + y)};
}
std::vector<size_t> GeometryEditor::CropPolygon() {
    auto &input = GetPoints();
    auto fmvp = camera_->GetProjectionMatrix() * camera_->GetViewMatrix();
    auto mvp = fmvp.cast<double>();
    auto vp = scene_->GetView()->GetViewport();  // x,y,w,h
    double half_width = (double)vp[2] * 0.5;
    double half_height = (double)vp[3] * 0.5;
    std::vector<Eigen::Vector2d> sels;
    for (auto &sel : selection_) {
        sels.emplace_back(sel.x(), vp[3] - sel.y());
    }

    std::vector<size_t> output_index;
#pragma omp parallel for
    for (size_t k = 0; k < input.size(); k++) {
        std::vector<double> nodes;
        const auto &point = input[k];
        auto pos = mvp * Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        if (pos(3) != 0.0) {
            pos /= pos(3);
            double x = (pos(0) + 1.0) * half_width;
            double y = (pos(1) + 1.0) * half_height;
            for (size_t i = 0; i < sels.size(); i++) {
                size_t j = (i + 1) % sels.size();
                auto &curr = sels[i];
                auto &next = sels[j];
                if ((curr(1) < y && next(1) >= y) ||
                    (next(1) < y && curr(1) >= y)) {
                    nodes.push_back(curr(0) + (y - curr(1)) /
                                                      (next(1) - curr(1)) *
                                                      (next(0) - curr(0)));
                }
            }
            std::sort(nodes.begin(), nodes.end());
            auto loc = std::lower_bound(nodes.begin(), nodes.end(), x);
            if (std::distance(nodes.begin(), loc) % 2 == 1) {
#pragma omp critical
                output_index.push_back(k);
            }
        }
    }

    return output_index;
}

std::vector<size_t> GeometryEditor::CropRectangle() {
    auto &input = GetPoints();
    auto fmvp = camera_->GetProjectionMatrix() * camera_->GetViewMatrix();
    auto mvp = fmvp.cast<double>();
    auto vp = scene_->GetView()->GetViewport();  // x,y,w,h
    double half_width = (double)vp[2] * 0.5;
    double half_height = (double)vp[3] * 0.5;
    std::vector<Eigen::Vector2d> sels;
    for (auto &sel : selection_) {
        sels.emplace_back(sel.x(), vp[3] - sel.y());
    }

    auto minX = std::min(sels[0].x(), sels[1].x());
    auto maxX = std::max(sels[0].x(), sels[1].x());
    auto minY = std::min(sels[0].y(), sels[1].y());
    auto maxY = std::max(sels[0].y(), sels[1].y());
    std::vector<size_t> output_index;
#pragma omp parallel for
    for (size_t k = 0; k < input.size(); k++) {
        const auto &point = input[k];
        auto pos = mvp * Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        if (pos(3) != 0.0) {
            pos /= pos(3);
            double x = (pos(0) + 1.0) * half_width;
            double y = (pos(1) + 1.0) * half_height;
            if (x >= minX && x <= maxX && y >= minY && y <= maxY) {
#pragma omp critical
                output_index.push_back(k);
            }
        }
    }

    return output_index;
}
std::vector<size_t> GeometryEditor::CropCircle() {
    auto &input = GetPoints();
    auto fmvp = camera_->GetProjectionMatrix() * camera_->GetViewMatrix();
    auto mvp = fmvp.cast<double>();
    auto vp = scene_->GetView()->GetViewport();  // x,y,w,h
    double half_width = (double)vp[2] * 0.5;
    double half_height = (double)vp[3] * 0.5;
    std::vector<Eigen::Vector2d> sels;
    for (auto &sel : selection_) {
        sels.emplace_back(sel.x(), vp[3] - sel.y());
    }
    auto &center = sels[0];
    Eigen::Vector2d d{sels[1].x() - center.x(), sels[1].y() - center.y()};
    auto s = d.squaredNorm();
    auto radius = d.norm();
    auto minX = center.x() - radius;
    auto maxX = center.x() + radius;
    auto minY = center.y() - radius;
    auto maxY = center.y() + radius;
    std::vector<size_t> output_index;

#pragma omp parallel for
    for (size_t k = 0; k < input.size(); k++) {
        const auto &point = input[k];
        auto pos = mvp * Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        if (pos(3) != 0.0) {
            pos /= pos(3);
            double x = (pos(0) + 1.0) * half_width;
            double y = (pos(1) + 1.0) * half_height;
            if (x >= minX && x <= maxX && y >= minY && y <= maxY) {
                Eigen::Vector2d p(x - center.x(), y - center.y());
                if (p.squaredNorm() < s) {
#pragma omp critical
                    output_index.push_back(k);
                }
            }
        }
    }

    return output_index;
}
}  // namespace gui
}  // namespace visualization
}  // namespace open3d
