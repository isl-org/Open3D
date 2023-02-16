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

#include "open3d/visualization/gui/PickPointsInteractor.h"

#include <unordered_map>
#include <unordered_set>

#include "open3d/geometry/Image.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/gui/Events.h"
#include "open3d/visualization/rendering/MaterialRecord.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/rendering/View.h"

#define WANT_DEBUG_IMAGE 0

#if WANT_DEBUG_IMAGE
#include "open3d/io/ImageIO.h"
#endif  // WANT_DEBUG_IMAGE

namespace open3d {
namespace visualization {
namespace gui {

namespace {
// Background color is white, so that index 0 can be black
static const Eigen::Vector4f kBackgroundColor = {1.0f, 1.0f, 1.0f, 1.0f};
static const std::string kSelectablePointsName = "__selectable_points";
// The maximum pickable point is one less than FFFFFF, because that would
// be white, which is the color of the background.
// static const unsigned int kNoIndex = 0x00ffffff;  // unused, but real
static const unsigned int kMeshIndex = 0x00fffffe;
static const unsigned int kMaxPickableIndex = 0x00fffffd;

inline bool IsValidIndex(uint32_t idx) { return (idx <= kMaxPickableIndex); }

Eigen::Vector3d CalcIndexColor(uint32_t idx) {
    const double red = double((idx & 0x00ff0000) >> 16) / 255.0;
    const double green = double((idx & 0x0000ff00) >> 8) / 255.0;
    const double blue = double((idx & 0x000000ff)) / 255.0;
    return {red, green, blue};
}

Eigen::Vector3d SetColorForIndex(uint32_t idx) {
    return CalcIndexColor(std::min(kMaxPickableIndex, idx));
}

uint32_t GetIndexForColor(geometry::Image *image, int x, int y) {
    uint8_t *rgb = image->PointerAt<uint8_t>(x, y, 0);
    const unsigned int red = (static_cast<unsigned int>(rgb[0]) << 16);
    const unsigned int green = (static_cast<unsigned int>(rgb[1]) << 8);
    const unsigned int blue = (static_cast<unsigned int>(rgb[2]));
    return (red | green | blue);
}

}  // namespace

// ----------------------------------------------------------------------------
class SelectionIndexLookup {
private:
    struct Obj {
        std::string name;
        size_t start_index;

        Obj(const std::string &n, size_t start) : name(n), start_index(start){};
        bool IsValid() const { return !name.empty(); }
    };

public:
    void Clear() { objects_.clear(); }

    // start_index must be larger than all previously added items
    void Add(const std::string &name, size_t start_index) {
        if (!objects_.empty() && objects_.back().start_index >= start_index) {
            utility::LogError(
                    "start_index {} must be larger than all previously added "
                    "objects {}.",
                    start_index, objects_.back().start_index);
        }
        objects_.emplace_back(name, start_index);
        if (objects_[0].start_index != 0) {
            utility::LogError(
                    "The first object's start_index must be 0, but got {}.",
                    objects_[0].start_index);
        }
    }

    const Obj &ObjectForIndex(size_t index) {
        if (objects_.size() == 1) {
            return objects_[0];
        } else {
            auto next = std::upper_bound(objects_.begin(), objects_.end(),
                                         index, [](size_t value, const Obj &o) {
                                             return value < o.start_index;
                                         });
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
PickPointsInteractor::PickPointsInteractor(rendering::Open3DScene *scene,
                                           rendering::Camera *camera) {
    scene_ = scene;
    camera_ = camera;
    picking_scene_ =
            std::make_shared<rendering::Open3DScene>(scene->GetRenderer());

    picking_scene_->SetDownsampleThreshold(SIZE_MAX);  // don't downsample!
    picking_scene_->SetBackground(kBackgroundColor);

    picking_scene_->GetView()->ConfigureForColorPicking();
}

PickPointsInteractor::~PickPointsInteractor() { delete lookup_; }

void PickPointsInteractor::SetPointSize(int px) {
    point_size_ = px;
    if (!points_.empty()) {
        auto mat = MakeMaterial();
        picking_scene_->GetScene()->OverrideMaterial(kSelectablePointsName,
                                                     mat);
    }
}

void PickPointsInteractor::SetPickableGeometry(
        const std::vector<SceneWidget::PickableGeometry> &geometry) {
    delete lookup_;
    lookup_ = new SelectionIndexLookup();

    picking_scene_->ClearGeometry();
    SetNeedsRedraw();

    // Record the points (for selection), and add a depth-write copy of any
    // TriangleMesh so that occluded points are not selected.
    points_.clear();
    for (auto &pg : geometry) {
        lookup_->Add(pg.name, points_.size());

        auto cloud = dynamic_cast<const geometry::PointCloud *>(pg.geometry);
        auto tcloud =
                dynamic_cast<const t::geometry::PointCloud *>(pg.tgeometry);
        auto mesh = dynamic_cast<const geometry::TriangleMesh *>(pg.geometry);
        auto tmesh =
                dynamic_cast<const t::geometry::TriangleMesh *>(pg.tgeometry);
        if (cloud) {
            points_.insert(points_.end(), cloud->points_.begin(),
                           cloud->points_.end());
        } else if (mesh) {
            points_.insert(points_.end(), mesh->vertices_.begin(),
                           mesh->vertices_.end());
        } else if (tcloud || tmesh) {
            const auto &tpoints = (tcloud ? tcloud->GetPointPositions()
                                          : tmesh->GetVertexPositions());
            const size_t n = tpoints.NumElements();
            float *pts = (float *)tpoints.GetDataPtr();
            points_.reserve(points_.size() + n);
            for (size_t i = 0; i < n; i += 3) {
                points_.emplace_back(double(pts[i]), double(pts[i + 1]),
                                     double(pts[i + 2]));
            }
        }

        if (mesh || tmesh) {
            // If we draw unlit with the background color, then if the mesh is
            // drawn before the picking points the end effect is to just write
            // to the depth buffer,and if we draw after the points then we paint
            // over the occluded points. We paint with a special "mesh index"
            // so that we can to enhanced picking if we hit a mesh index.
            auto mesh_color = CalcIndexColor(kMeshIndex);
            rendering::MaterialRecord mat;
            mat.shader = "unlitSolidColor";  // ignore any vertex colors!
            mat.base_color = {float(mesh_color.x()), float(mesh_color.y()),
                              float(mesh_color.z()), 1.0f};
            mat.sRGB_color = false;
            if (mesh) {
                picking_scene_->AddGeometry(pg.name, mesh, mat);
            } else {
                utility::LogWarning(
                        "PickPointsInteractor::SetPickableGeometry(): "
                        "Open3DScene cannot add a t::geometry::TriangleMesh, "
                        "so points on the back side of the mesh '{}', will be "
                        "pickable",
                        pg.name);
                // picking_scene_->AddGeometry(pg.name, tmesh, mat);
            }
        }
    }
    // add safety but invalid obj
    lookup_->Add("", points_.size());

    if (points_.size() > kMaxPickableIndex) {
        utility::LogWarning(
                "Can only select from a maximumum of {} points; given {}",
                kMaxPickableIndex, points_.size());
    }

    if (!points_.empty()) {  // Filament panics if an object has zero vertices
        auto cloud = std::make_shared<geometry::PointCloud>(points_);
        cloud->colors_.reserve(points_.size());
        for (size_t i = 0; i < cloud->points_.size(); ++i) {
            cloud->colors_.emplace_back(SetColorForIndex(uint32_t(i)));
        }

        auto mat = MakeMaterial();
        picking_scene_->AddGeometry(kSelectablePointsName, cloud.get(), mat);
        picking_scene_->GetScene()->GeometryShadows(kSelectablePointsName,
                                                    false, false);
    }
}

void PickPointsInteractor::SetNeedsRedraw() { dirty_ = true; }

rendering::MatrixInteractorLogic &PickPointsInteractor::GetMatrixInteractor() {
    return matrix_logic_;
}

void PickPointsInteractor::SetOnPointsPicked(
        std::function<void(
                const std::map<std::string,
                               std::vector<std::pair<size_t, Eigen::Vector3d>>>
                        &,
                int)> f) {
    on_picked_ = f;
}

void PickPointsInteractor::SetOnUIChanged(
        std::function<void(const std::vector<Eigen::Vector2i> &)> on_ui) {
    on_ui_changed_ = on_ui;
}

void PickPointsInteractor::SetOnStartedPolygonPicking(
        std::function<void()> on_poly_pick) {
    on_started_poly_pick_ = on_poly_pick;
}

void PickPointsInteractor::Mouse(const MouseEvent &e) {
    if (e.type == MouseEvent::BUTTON_UP) {
        if (e.modifiers & int(KeyModifier::ALT)) {
            if (pending_.empty() || pending_.back().keymods == 0) {
                pending_.push({{gui::Point(e.x, e.y)}, int(KeyModifier::ALT)});
                if (on_ui_changed_) {
                    on_ui_changed_({});
                }
            } else {
                pending_.back().polygon.push_back(gui::Point(e.x, e.y));
                if (on_started_poly_pick_) {
                    on_started_poly_pick_();
                }
                if (on_ui_changed_) {
                    std::vector<Eigen::Vector2i> lines;
                    auto &polygon = pending_.back().polygon;
                    for (size_t i = 1; i < polygon.size(); ++i) {
                        auto &p0 = polygon[i - 1];
                        auto &p1 = polygon[i];
                        lines.push_back({p0.x, p0.y});
                        lines.push_back({p1.x, p1.y});
                    }
                    lines.push_back({polygon.back().x, polygon.back().y});
                    lines.push_back({polygon[0].x, polygon[0].y});
                    on_ui_changed_(lines);
                }
            }
        } else {
            pending_.push({{gui::Point(e.x, e.y)}, 0});
            DoPick();
        }
    }
}

void PickPointsInteractor::Key(const KeyEvent &e) {
    if (e.type == KeyEvent::UP) {
        if (e.key == KEY_ESCAPE) {
            ClearPick();
        }
    }
}

void PickPointsInteractor::DoPick() {
    if (dirty_) {
        SetNeedsRedraw();
        auto *view = picking_scene_->GetView();
        view->SetViewport(0, 0,  // in case scene widget changed size
                          matrix_logic_.GetViewWidth(),
                          matrix_logic_.GetViewHeight());
        view->GetCamera()->CopyFrom(camera_);
        picking_scene_->GetRenderer().RenderToImage(
                view, picking_scene_->GetScene(),
                [this](std::shared_ptr<geometry::Image> img) {
#if WANT_DEBUG_IMAGE
                    std::cout << "[debug] Writing pick image to "
                              << "debug.png" << std::endl;
                    io::WriteImage("debug.png", *img);
#endif  // WANT_DEBUG_IMAGE
                    this->OnPickImageDone(img);
                });
    } else {
        OnPickImageDone(pick_image_);
    }
}

void PickPointsInteractor::ClearPick() {
    while (!pending_.empty()) {
        pending_.pop();
    }
    if (on_ui_changed_) {
        on_ui_changed_({});
    }
    SetNeedsRedraw();
}

rendering::MaterialRecord PickPointsInteractor::MakeMaterial() {
    rendering::MaterialRecord mat;
    mat.shader = "unlitPolygonOffset";
    mat.point_size = float(point_size_);
    // We are not tonemapping, so src colors are RGB. This prevents the colors
    // being converted from sRGB -> linear like normal.
    mat.sRGB_color = false;
    return mat;
}

void PickPointsInteractor::OnPickImageDone(
        std::shared_ptr<geometry::Image> img) {
    if (dirty_) {
        pick_image_ = img;
        dirty_ = false;
    }

    if (on_ui_changed_) {
        on_ui_changed_({});
    }

    std::map<std::string, std::vector<std::pair<size_t, Eigen::Vector3d>>>
            indices;
    while (!pending_.empty()) {
        PickInfo &info = pending_.back();
        auto *img = pick_image_.get();
        indices.clear();
        if (info.polygon.size() == 1) {
            const int x0 = info.polygon[0].x;
            const int y0 = info.polygon[0].y;
            struct Score {  // this is a struct to force a default value
                float score = 0;
            };
            std::unordered_map<unsigned int, Score> candidates;
            auto clicked_idx = GetIndexForColor(img, x0, y0);
            int radius;
            // HACK: the color for kMeshIndex doesn't come back quite right.
            //       We shouldn't need to check if the index is out of range,
            //       but it does work.
            if (clicked_idx == kMeshIndex || clicked_idx >= points_.size()) {
                // We hit the middle of a triangle, try to find a nearby point
                radius = 5 * point_size_;
            } else {
                // We either hit a point or an empty spot, so use a smaller
                // radius. It looks weird to click on nothing in a point cloud
                // and have a point get selected unless the point is really
                // close.
                radius = 2 * point_size_;
            }
            for (int y = y0 - radius; y < y0 + radius; ++y) {
                for (int x = x0 - radius; x < x0 + radius; ++x) {
                    unsigned int idx = GetIndexForColor(img, x, y);
                    if (IsValidIndex(idx) && idx < points_.size()) {
                        float dist = std::sqrt(float((x - x0) * (x - x0) +
                                                     (y - y0) * (y - y0)));
                        candidates[idx].score += radius - dist;
                    }
                }
            }
            if (!candidates.empty()) {
                // Note that scores are (radius - dist), and since we take from
                // a square pattern, a score can be negative. And multiple
                // pixels of a point scoring negatively can make the negative up
                // to -point_size^2.
                float best_score = -1e30f;
                unsigned int best_idx = (unsigned int)-1;
                for (auto &idx_score : candidates) {
                    if (idx_score.second.score > best_score) {
                        best_score = idx_score.second.score;
                        best_idx = idx_score.first;
                    }
                }
                auto &o = lookup_->ObjectForIndex(best_idx);
                if (o.IsValid()) {
                    size_t obj_idx = best_idx - o.start_index;
                    indices[o.name].push_back(
                            std::pair<size_t, Eigen::Vector3d>(
                                    obj_idx, points_[best_idx]));
                }
            }
        } else {
            // Use polygon fill algorithm to find the pixels that need to be
            // checked.
            // Good test cases:  ______________             /|
            //                  |             /    |\      / |
            //                  |            /     |  \   /  |
            //                  |   /\      /      |    \/   |
            //                  |  /  \    /      |          |
            //                  | /    \  /       |     _____|
            //                  |/      \/        |____/
            std::unordered_set<unsigned int> raw_indices;

            // Find the min/max y, so we can avoid excess looping.
            int minY = 1000000, maxY = -1000000;
            for (auto &p : info.polygon) {
                minY = std::min(minY, p.y);
                maxY = std::max(maxY, p.y);
            }
            // Duplicate the first point so for easy indexing
            info.polygon.push_back(info.polygon[0]);
            // Precalculate m and b (of y = mx + b)
            const double kInf = 1e18;
            std::vector<double> m, b;
            m.reserve(info.polygon.size() - 1);
            b.reserve(info.polygon.size() - 1);
            for (size_t i = 1; i < info.polygon.size(); ++i) {
                int m_denom = info.polygon[i].x - info.polygon[i - 1].x;
                if (m_denom == 0) {  // vertical line (x doesn't change)
                    m.push_back(kInf);
                    b.push_back(0.0);
                    continue;
                }
                m.push_back(double(info.polygon[i].y - info.polygon[i - 1].y) /
                            double(m_denom));
                if (m.back() == 0.0) {  // horiz line (y doesn't change)
                    b.push_back(info.polygon[i].y);
                } else {
                    b.push_back(info.polygon[i].y -
                                m.back() * info.polygon[i].x);
                }
            }
            // Loop through the rows of the polygon.
            std::vector<bool> is_vert_corner(info.polygon.size(), false);
            for (size_t i = 0; i < info.polygon.size() - 1; ++i) {
                int prev = i - 1;
                if (prev < 0) {
                    prev = info.polygon.size() - 2;
                }
                int next = i + 1;
                int lastY = info.polygon[prev].y;
                int thisY = info.polygon[i].y;
                int nextY = info.polygon[next].y;
                if ((thisY > lastY && thisY > nextY) ||
                    (thisY < lastY && thisY < nextY)) {
                    is_vert_corner[i] = true;
                }
            }
            is_vert_corner.back() = is_vert_corner[0];
            std::unordered_set<int> intersectionsX;
            std::vector<int> sortedX;
            intersectionsX.reserve(32);
            sortedX.reserve(32);
            for (int y = minY; y <= maxY; ++y) {
                for (size_t i = 0; i < m.size(); ++i) {
                    if ((y < info.polygon[i].y && y < info.polygon[i + 1].y) ||
                        (y > info.polygon[i].y && y > info.polygon[i + 1].y)) {
                        continue;
                    }
                    if (m[i] == 0.0) {  // horizontal
                        intersectionsX.insert({info.polygon[i].x});
                        intersectionsX.insert({info.polygon[i + 1].x});
                    } else if (m[i] == kInf) {  // vertical
                        bool is_corner = (y == info.polygon[i].y);
                        intersectionsX.insert({info.polygon[i].x});
                        if (is_corner) {
                            intersectionsX.insert({info.polygon[i].x});
                        }
                    } else {
                        double x = (double(y) - b[i]) / m[i];
                        bool is_corner0 =
                                (y == info.polygon[i].y &&
                                 std::abs(x - double(info.polygon[i].x)) < 0.5);
                        bool is_corner1 =
                                (y == info.polygon[i + 1].y &&
                                 std::abs(x - double(info.polygon[i + 1].x)) <
                                         0.5);
                        if ((is_corner0 && is_vert_corner[i]) ||
                            (is_corner1 && is_vert_corner[i + 1])) {
                            // We hit the corner, don't add, otherwise we will
                            // get half a segment.
                        } else {
                            intersectionsX.insert(int(std::round(x)));
                        }
                    }
                }
                for (auto x : intersectionsX) {
                    sortedX.push_back(x);
                }
                std::sort(sortedX.begin(), sortedX.end());

                // sortedX contains the horizontal line segment(s). This should
                // be an even number, otherwise there is a problem. (Probably
                // a corner got included)
                if (sortedX.size() % 2 == 1) {
                    std::stringstream s;
                    for (size_t i = 0; i < info.polygon.size() - 1; ++i) {
                        s << "(" << info.polygon[i].x << ", "
                          << info.polygon[i].y << ") ";
                    }
                    utility::LogWarning(
                            "Internal error: Odd number of points for row "
                            "segments (should be even).");
                    utility::LogWarning("Polygon is: {}", s.str());
                    s.str("");
                    s << "{ ";
                    for (size_t i = 0; i < sortedX.size(); ++i) {
                        s << sortedX[i] << " ";
                    }
                    s << "}";
                    utility::LogWarning("y: {}, sortedX: {}", y, s.str());
                    // Recover: this is likely to give the wrong result, but
                    // better than the alternative of crashing.
                    sortedX.push_back(sortedX.back());
                }

                // "Fill" the pixels on this row
                for (size_t i = 0; i < sortedX.size(); i += 2) {
                    int startX = sortedX[i];
                    int endX = sortedX[i + 1];
                    for (int x = startX; x <= endX; ++x) {
                        unsigned int idx = GetIndexForColor(img, x, y);
                        if (IsValidIndex(idx) && idx < points_.size()) {
                            raw_indices.insert(idx);
                        }
                    }
                }
                intersectionsX.clear();
                sortedX.clear();
            }
            // Now add everything that was "filled"
            for (auto idx : raw_indices) {
                auto &o = lookup_->ObjectForIndex(idx);
                if (o.IsValid()) {
                    size_t obj_idx = idx - o.start_index;
                    indices[o.name].push_back(
                            std::pair<size_t, Eigen::Vector3d>(obj_idx,
                                                               points_[idx]));
                }
            }
        }

        pending_.pop();

        if (on_picked_ && !indices.empty()) {
            on_picked_(indices, info.keymods);
        }
    }
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
