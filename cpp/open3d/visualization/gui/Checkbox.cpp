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

#include "open3d/visualization/gui/Checkbox.h"

#include <imgui.h>

#include <cmath>
#include <string>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

struct Checkbox::Impl {
    std::string name_;
    bool is_checked_ = false;
    std::function<void(bool)> on_checked_;
};

Checkbox::Checkbox(const char* name) : impl_(new Checkbox::Impl()) {
    impl_->name_ = name;
}

Checkbox::~Checkbox() {}

bool Checkbox::IsChecked() const { return impl_->is_checked_; }

void Checkbox::SetChecked(bool checked) { impl_->is_checked_ = checked; }

void Checkbox::SetOnChecked(std::function<void(bool)> on_checked) {
    impl_->on_checked_ = on_checked;
}

Size Checkbox::CalcPreferredSize(const Theme& theme) const {
    auto em = ImGui::GetTextLineHeight();
    auto padding = ImGui::GetStyle().FramePadding;
    auto text_size = ImGui::GetFont()->CalcTextSizeA(
            float(theme.font_size), 10000, 10000, impl_->name_.c_str());
    int height = int(std::ceil(em + 2.0f * padding.y));
    auto checkbox_width = height + padding.x;
    return Size(int(checkbox_width + std::ceil(text_size.x + 2.0f * padding.x)),
                height);
}

Widget::DrawResult Checkbox::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorScreenPos(ImVec2(float(frame.x), float(frame.y)));
    auto result = Widget::DrawResult::NONE;

    // ImGUI doesn't offer styling specific to checkboxes other than the
    // color of the checkmark, so we need to adjust the colors ourselves.
    if (impl_->is_checked_) {
        ImGui::PushStyleColor(
                ImGuiCol_FrameBg,
                colorToImgui(context.theme.checkbox_background_on_color));
        ImGui::PushStyleColor(
                ImGuiCol_FrameBgHovered,
                colorToImgui(context.theme.checkbox_background_hover_on_color));
    } else {
        ImGui::PushStyleColor(
                ImGuiCol_FrameBg,
                colorToImgui(context.theme.checkbox_background_off_color));
        ImGui::PushStyleColor(
                ImGuiCol_FrameBgHovered,
                colorToImgui(
                        context.theme.checkbox_background_hover_off_color));
    }

    DrawImGuiPushEnabledState();
    ImGui::PushItemWidth(float(GetFrame().width));
    if (ImGui::Checkbox(impl_->name_.c_str(), &impl_->is_checked_)) {
        if (impl_->on_checked_) {
            impl_->on_checked_(impl_->is_checked_);
        }
        result = Widget::DrawResult::REDRAW;
    }
    ImGui::PopItemWidth();
    DrawImGuiPopEnabledState();

    ImGui::PopStyleColor(2);

    return result;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
