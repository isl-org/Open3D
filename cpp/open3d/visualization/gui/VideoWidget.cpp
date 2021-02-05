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

#include "open3d/visualization/gui/VideoWidget.h"

#include <imgui.h>

#include "open3d/geometry/Image.h"
#include "open3d/visualization/gui/Events.h"
#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/UIImage.h"
#include "open3d/visualization/rendering/VideoProvider.h"

namespace open3d {
namespace visualization {
namespace gui {

struct VideoWidget::Impl {
    bool is_playing_ = true;
    std::shared_ptr<rendering::VideoProvider> video_;
    double current_time_;
    std::shared_ptr<UIImage> current_frame_;
};

VideoWidget::VideoWidget(std::shared_ptr<rendering::VideoProvider> video)
    : impl_(new VideoWidget::Impl()) {
    SetVideoProvider(video);
}

VideoWidget::~VideoWidget() {}

std::shared_ptr<rendering::VideoProvider> VideoWidget::GetVideoProvider() {
    return impl_->video_;
}

void VideoWidget::SetVideoProvider(
        std::shared_ptr<rendering::VideoProvider> video) {
    impl_->video_ = video;
    impl_->current_time_ = 0.0;
    impl_->video_->SetTime(0.0);
    impl_->current_frame_ =
            std::make_shared<UIImage>(impl_->video_->GetFrame());
    impl_->current_frame_->SetScaling(UIImage::Scaling::ASPECT);
}

bool VideoWidget::GetIsPlaying() const { return impl_->is_playing_; }

void VideoWidget::SetIsPlaying(bool playing) { impl_->is_playing_ = playing; }

Widget::DrawResult VideoWidget::Tick(const TickEvent& e) {
    if (!impl_->is_playing_) {
        return Widget::DrawResult::NONE;
    }

    impl_->current_time_ += e.dt;
    if (impl_->video_->SetTime(impl_->current_time_) ==
        rendering::VideoProvider::UpdateResult::NEEDS_REDRAW) {
        impl_->current_frame_->UpdateImage(impl_->video_->GetFrame());
        return Widget::DrawResult::REDRAW;
    } else {
        return Widget::DrawResult::NONE;
    }
}

Widget::DrawResult VideoWidget::Draw(const DrawContext& context) {
    UIImage::DrawParams params;  // .texture defaults to kBad handle
    if (impl_->current_frame_) {
        params = impl_->current_frame_->CalcDrawParams(context.renderer,
                                                       GetFrame());
    }

    ImTextureID image_id =
            reinterpret_cast<ImTextureID>(params.texture.GetId());
    ImGui::SetCursorScreenPos(ImVec2(params.pos_x, params.pos_y));
    ImGui::Image(image_id, ImVec2(params.width, params.height),
                 ImVec2(params.u0, params.v0), ImVec2(params.u1, params.v1));

    return Widget::DrawResult::NONE;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
