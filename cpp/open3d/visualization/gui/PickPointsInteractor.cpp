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

#include "open3d/visualization/gui/PickPointsInteractor.h"

#include "open3d/geometry/Image.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/utility/Console.h"
#include "open3d/visualization/gui/Events.h"
#include "open3d/visualization/rendering/Material.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/rendering/View.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
static const std::string kSelectablePointsName = "__selectable_points";
// The maximum pickable point is one less than FFFFFF, because that would
// be white, which is the color of the background.
static const unsigned int kNoIndex = 0x00ffffff;
static const unsigned int kMaxPickableIndex = 0x00fffffe;

Eigen::Vector3d SetColorForIndex(uint32_t idx) {
    idx = std::min(kMaxPickableIndex, idx);
    const double red = double((idx & 0x00ff0000) >> 16) / 255.0;
    const double green = double((idx & 0x0000ff00) >> 8) / 255.0;
    const double blue = double((idx & 0x000000ff)) / 255.0;
    return {red, green, blue};
}

uint32_t GetIndexForColor(geometry::Image *image, int x, int y) {
    uint8_t *rgb = image->PointerAt<uint8_t>(x, y, 0);
    const unsigned int red = (static_cast<unsigned int>(rgb[0]) << 16);
    const unsigned int green = (static_cast<unsigned int>(rgb[1]) << 8);
    const unsigned int blue = (static_cast<unsigned int>(rgb[2]));
    return (red | green | blue);
}

}  // namespace

PickPointsInteractor::PickPointsInteractor(rendering::Open3DScene *scene,
                                           rendering::Camera *camera) {
    scene_ = scene;
    camera_ = camera;
    picking_scene_ =
            std::make_shared<rendering::Open3DScene>(scene->GetRenderer());

    picking_scene_->SetDownsampleThreshold(SIZE_MAX);  // don't downsample!
    // Background color is white, so that index 0 can be black
    picking_scene_->SetBackgroundColor({1.0f, 1.0f, 1.0f, 1.0f});

    picking_scene_->GetView()->ConfigureForColorPicking();
}

PickPointsInteractor::~PickPointsInteractor() {}

void PickPointsInteractor::SetPointSize(int px) {
    point_size_ = px;
    if (!pickable_points_.empty()) {
        auto mat = MakeMaterial();
        picking_scene_->GetScene()->OverrideMaterial(kSelectablePointsName,
                                                     mat);
    }
}

void PickPointsInteractor::SetPickablePoints(
        const std::vector<Eigen::Vector3d> &points) {
    if (points.size() > kMaxPickableIndex) {
        utility::LogWarning(
                "Can only select from a maximumum of {} points; given {}",
                kMaxPickableIndex, points.size());
    }

    pickable_points_ = points;
    SetNeedsRedraw();

    picking_scene_->RemoveGeometry(kSelectablePointsName);

    if (!points.empty()) {  // Filament panics if an object has zero vertices
        auto cloud = std::make_shared<geometry::PointCloud>(points);
        cloud->colors_.reserve(points.size());
        for (size_t i = 0; i < cloud->points_.size(); ++i) {
            cloud->colors_.emplace_back(SetColorForIndex(uint32_t(i)));
        }

        auto mat = MakeMaterial();
        picking_scene_->AddGeometry(kSelectablePointsName, cloud, mat);
        picking_scene_->GetScene()->GeometryShadows(kSelectablePointsName,
                                                    false, false);
    }
}

void PickPointsInteractor::SetNeedsRedraw() {
    dirty_ = true;
    // std::queue::clear() seems to not exist
    while (!pending_.empty()) {
        pending_.pop();
    }
}

rendering::MatrixInteractorLogic &PickPointsInteractor::GetMatrixInteractor() {
    return matrix_logic_;
}

void PickPointsInteractor::SetOnPointsPicked(
        std::function<void(const std::vector<size_t> &, int)> f) {
    on_picked_ = f;
}

void PickPointsInteractor::Mouse(const MouseEvent &e) {
    if (e.type == MouseEvent::BUTTON_UP) {
        gui::Rect pick_rect(e.x, e.y, 1, 1);
        if (dirty_) {
            SetNeedsRedraw();  // note: clears pending_
            pending_.push({pick_rect, e.modifiers});
            auto *view = picking_scene_->GetView();
            view->SetViewport(0, 0,  // in case scene widget changed size
                              matrix_logic_.GetViewWidth(),
                              matrix_logic_.GetViewHeight());
            view->GetCamera()->CopyFrom(camera_);
            picking_scene_->GetRenderer().RenderToImage(
                    view, picking_scene_->GetScene(),
                    [this](std::shared_ptr<geometry::Image> img) {
                        this->OnPickImageDone(img);
                        // io::WriteImage("/tmp/debug.png", *img); // debugging
                    });
        } else {
            pending_.push({pick_rect, e.modifiers});
            OnPickImageDone(pick_image_);
        }
    }
}

// TODO: do we need this?
void PickPointsInteractor::Key(const KeyEvent &e) {}

rendering::Material PickPointsInteractor::MakeMaterial() {
    rendering::Material mat;
    mat.shader = "defaultUnlit";
    mat.point_size = float(point_size_);
    // We are not tonemapping, so src colors are RGB. This prevents the colors
    // being converetd from sRGB -> linear like normal.
    mat.sRGB_color = false;
    return mat;
}

void PickPointsInteractor::OnPickImageDone(
        std::shared_ptr<geometry::Image> img) {
    if (dirty_) {
        pick_image_ = img;
        dirty_ = false;
    }

    std::vector<size_t> indices;
    while (!pending_.empty()) {
        PickInfo &info = pending_.back();
        const int x0 = info.rect.x;
        const int x1 = info.rect.GetRight();
        const int y0 = info.rect.y;
        const int y1 = info.rect.GetBottom();
        auto *img = pick_image_.get();
        indices.clear();
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                unsigned int idx = GetIndexForColor(img, x, y);
                if (idx < kNoIndex) {
                    indices.push_back(idx);
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
