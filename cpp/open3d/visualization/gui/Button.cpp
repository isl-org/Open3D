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

#include "open3d/visualization/gui/Button.h"

#include <imgui.h>

#include <cmath>
#include <string>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
static int g_next_button_id = 1;
}  // namespace

struct Button::Impl {
    std::string id_;
    std::string title_;
    std::shared_ptr<UIImage> image_;
    float padding_horizontal_em_ = 0.5f;
    float padding_vertical_em_ = 0.5f;
    bool is_toggleable_ = false;
    bool is_on_ = false;
    std::function<void()> on_clicked_;
};

Button::Button(const char* title) : impl_(new Button::Impl()) {
    impl_->id_ = std::string("##button") + std::to_string(g_next_button_id++);
    impl_->title_ = title;
}

Button::Button(std::shared_ptr<UIImage> image) : impl_(new Button::Impl()) {
    impl_->image_ = image;
}

Button::~Button() {}

const char* Button::GetText() const { return impl_->title_.c_str(); }

void Button::SetText(const char* text) { impl_->title_ = text; }

float Button::GetHorizontalPaddingEm() const {
    return impl_->padding_horizontal_em_;
}

float Button::GetVerticalPaddingEm() const {
    return impl_->padding_vertical_em_;
}

void Button::SetPaddingEm(float horiz_ems, float vert_ems) {
    impl_->padding_horizontal_em_ = horiz_ems;
    impl_->padding_vertical_em_ = vert_ems;
}

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

Size Button::CalcPreferredSize(const LayoutContext& context,
                               const Constraints& constraints) const {
    auto em = float(context.theme.font_size);
    auto padding_horiz = int(std::ceil(impl_->padding_horizontal_em_ * em));
    auto padding_vert = int(std::ceil(impl_->padding_vertical_em_ * em));
    if (impl_->image_) {
        auto size = impl_->image_->CalcPreferredSize(context, constraints);
        return Size(size.width + 2 * padding_horiz,
                    size.height + 2 * padding_vert);
    } else {
        auto font = ImGui::GetFont();
        auto imguiVertPadding = ImGui::GetTextLineHeightWithSpacing() -
                                ImGui::GetTextLineHeight();
        auto size = font->CalcTextSizeA(float(context.theme.font_size),
                                        float(constraints.width), 10000,
                                        impl_->title_.c_str());
        // When ImGUI draws text, it draws text in a box of height
        // font_size + spacing. The padding on the bottom is essentially the
        // descender height, and the box height ends up giving a visual padding
        // of descender_height on the top and bottom. So while we only need to
        // 1 * imguiVertPadding on the bottom, we need to add 2x on the sides.
        // Note that padding of 0 doesn't actually produce a padding of zero,
        // because that would look horrible. (And also because if we do that,
        // ImGUI will position the text so that the descender is cut off,
        // because it is assuming that it gets a little extra on the bottom.
        // This looks really bad...)
        return Size(
                int(std::ceil(size.x + 2.0f + imguiVertPadding)) +
                        2 * padding_horiz,
                int(std::ceil(ImGui::GetTextLineHeight() + imguiVertPadding)) +
                        2 * padding_vert);
    }
}

Widget::DrawResult Button::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    auto result = Widget::DrawResult::NONE;

    ImGui::SetCursorScreenPos(
            ImVec2(float(frame.x), float(frame.y) - ImGui::GetScrollY()));

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
    bool def_bg = IsDefaultBackgroundColor();
    if (!def_bg) {
        ImGui::PushStyleColor(ImGuiCol_Button,
                              colorToImgui(Widget::GetBackgroundColor()));
    }
    DrawImGuiPushEnabledState();
    bool pressed = false;
    if (impl_->image_) {
        auto params = impl_->image_->CalcDrawParams(context.renderer, frame);
        ImTextureID image_id =
                reinterpret_cast<ImTextureID>(params.texture.GetId());
        pressed = ImGui::ImageButton(
                image_id, ImVec2(params.width, params.height),
                ImVec2(params.u0, params.v0), ImVec2(params.u1, params.v1));
    } else {
        pressed = ImGui::Button(
                (impl_->title_ + impl_->id_).c_str(),
                ImVec2(float(GetFrame().width), float(GetFrame().height)));
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
    DrawImGuiTooltip();
    if (!def_bg) {
        ImGui::PopStyleColor(1);
    }
    if (was_on) {
        ImGui::PopStyleColor(4);
    }

    return result;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
