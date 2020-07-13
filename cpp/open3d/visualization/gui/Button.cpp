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

#include "open3d/visualization/gui/Button.h"

#include <imgui.h>
#include <cmath>
#include <string>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

struct Button::Impl {
    std::string title_;
    std::shared_ptr<UIImage> image_;
    bool is_toggleable_ = false;
    bool is_on_ = false;
    std::function<void()> on_clicked_;
};

Button::Button(const char* title) : impl_(new Button::Impl()) {
    impl_->title_ = title;
}

Button::Button(std::shared_ptr<UIImage> image) : impl_(new Button::Impl()) {
    impl_->image_ = image;
}

Button::~Button() {}

bool Button::GetIsToggleable() const { return impl_->is_toggleable_; }

void Button::SetToggleable(bool toggles) { impl_->is_toggleable_ = toggles; }

bool Button::GetIsOn() const { return impl_->is_on_; }

void Button::SetOn(bool is_on) {
    if (impl_->is_toggleable_) {
        impl_->is_on_ = is_on;
    }
}

void Button::SetOnClicked(std::function<void()> on_clicked) {
    impl_->on_clicked_ = on_clicked;
}

Size Button::CalcPreferredSize(const Theme& theme) const {
    if (impl_->image_) {
        auto size = impl_->image_->CalcPreferredSize(theme);
        return Size(size.width, size.height);
    } else {
        auto font = ImGui::GetFont();
        auto em = std::ceil(ImGui::GetTextLineHeight());
        auto size = font->CalcTextSizeA(theme.font_size, 10000, 10000,
                                        impl_->title_.c_str());
        return Size(std::ceil(size.x) + 2.0 * em, 2 * em);
    }
}

Widget::DrawResult Button::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    auto result = Widget::DrawResult::NONE;

    bool was_on = impl_->is_on_;
    if (was_on) {
        ImGui::PushStyleColor(ImGuiCol_Text,
                              colorToImgui(context.theme.button_on_text_color));
        ImGui::PushStyleColor(ImGuiCol_Button,
                              colorToImgui(context.theme.button_on_color));
        ImGui::PushStyleColor(
                ImGuiCol_ButtonHovered,
                colorToImgui(context.theme.button_on_hover_color));
        ImGui::PushStyleColor(
                ImGuiCol_ButtonActive,
                colorToImgui(context.theme.button_on_active_color));
    }
    DrawImGuiPushEnabledState();
    bool pressed = false;
    if (impl_->image_) {
        auto params = impl_->image_->CalcDrawParams(context.renderer, frame);
        ImTextureID image_id =
                reinterpret_cast<ImTextureID>(params.texture.GetId());
        ImGui::SetCursorPos(ImVec2(params.pos_x - context.uiOffsetX,
                                   params.pos_y - context.uiOffsetY));
        pressed = ImGui::ImageButton(
                image_id, ImVec2(params.width, params.height),
                ImVec2(params.u0, params.v0), ImVec2(params.u1, params.v1));
    } else {
        ImGui::SetCursorPos(ImVec2(frame.x - context.uiOffsetX,
                                   frame.y - context.uiOffsetY));
        pressed = ImGui::Button(impl_->title_.c_str(),
                                ImVec2(GetFrame().width, GetFrame().height));
    }
    if (pressed) {
        if (impl_->is_toggleable_) {
            impl_->is_on_ = !impl_->is_on_;
        }
        if (impl_->on_clicked_) {
            impl_->on_clicked_();
        }
        result = Widget::DrawResult::REDRAW;
    }
    DrawImGuiPopEnabledState();
    if (was_on) {
        ImGui::PopStyleColor(4);
    }

    return result;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
