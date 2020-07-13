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

#include "open3d/visualization/gui/ImageLabel.h"

#include <imgui.h>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

struct ImageLabel::Impl {
    std::shared_ptr<UIImage> image_;
};

ImageLabel::ImageLabel(const char* image_path) : impl_(new ImageLabel::Impl()) {
    impl_->image_ = std::make_shared<UIImage>(image_path);
}

ImageLabel::ImageLabel(visualization::rendering::TextureHandle texture_id,
                       float u0 /*= 0.0f*/,
                       float v0 /*= 0.0f*/,
                       float u1 /*= 1.0f*/,
                       float v1 /*= 1.0f*/)
    : impl_(new ImageLabel::Impl()) {
    impl_->image_ = std::make_shared<UIImage>(texture_id, u0, v0, u1, v1);
}

ImageLabel::ImageLabel(std::shared_ptr<UIImage> image)
    : impl_(new ImageLabel::Impl()) {
    impl_->image_ = image;
}

ImageLabel::~ImageLabel() {}

Size ImageLabel::CalcPreferredSize(const Theme& theme) const {
    Size pref;
    if (impl_->image_) {
        pref = impl_->image_->CalcPreferredSize(theme);
    }

    if (pref.width != 0 && pref.height != 0) {
        return pref;
    } else {
        return Size(5 * theme.font_size, 5 * theme.font_size);
    }
}

void ImageLabel::Layout(const Theme& theme) { Super::Layout(theme); }

Widget::DrawResult ImageLabel::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    UIImage::DrawParams params;  // .texture defaults to kBad handle
    if (impl_->image_) {
        params = impl_->image_->CalcDrawParams(context.renderer, frame);
    }

    if (params.texture != visualization::rendering::TextureHandle::kBad) {
        ImTextureID image_id =
                reinterpret_cast<ImTextureID>(params.texture.GetId());
        ImGui::SetCursorPos(ImVec2(params.pos_x - context.uiOffsetX,
                                   params.pos_y - context.uiOffsetY));
        ImGui::Image(image_id, ImVec2(params.width, params.height),
                     ImVec2(params.u0, params.v0),
                     ImVec2(params.u1, params.v1));
    } else {
        // Draw error message if we don't have an image, instead of
        // quietly failing or something.
        const char* error_text = "  Error\nloading\n image";
        Color fg(1.0, 1.0, 1.0);
        ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(frame.x, frame.y),
                ImVec2(frame.GetRight(), frame.GetBottom()),
                IM_COL32(255, 0, 0, 255));
        ImGui::GetWindowDrawList()->AddRect(
                ImVec2(frame.x, frame.y),
                ImVec2(frame.GetRight(), frame.GetBottom()),
                IM_COL32(255, 255, 255, 255));
        ImGui::PushStyleColor(ImGuiCol_Text, colorToImgui(fg));

        auto padding = ImGui::GetStyle().FramePadding;
        float wrap_width = frame.width - std::ceil(2.0f * padding.x);
        float wrapX = ImGui::GetCursorPos().x + wrap_width;
        auto* font = ImGui::GetFont();
        auto text_size = font->CalcTextSizeA(
                context.theme.font_size, wrap_width, wrap_width, error_text);
        float x = (float(frame.width) - text_size.x) / 2.0f;
        float y = (float(frame.height) - text_size.y) / 2.0f;

        ImGui::SetCursorPos(ImVec2(x + frame.x - context.uiOffsetX,
                                   y + frame.y - context.uiOffsetY));
        ImGui::PushTextWrapPos(wrapX);
        ImGui::TextWrapped("%s", error_text);
        ImGui::PopTextWrapPos();

        ImGui::PopStyleColor();
    }

    return Widget::DrawResult::NONE;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
