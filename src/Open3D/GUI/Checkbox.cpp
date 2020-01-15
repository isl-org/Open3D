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

#include "Checkbox.h"

#include "Theme.h"
#include "Util.h"

#include <imgui.h>

#include <cmath>
#include <string>

using namespace open3d::gui::util;

namespace open3d {
namespace gui {

struct Checkbox::Impl {
    std::string name;
    bool isChecked = false;
};

Checkbox::Checkbox(const char* name) : impl_(new Checkbox::Impl()) {
    impl_->name = name;
}

Checkbox::~Checkbox() {}

bool Checkbox::IsChecked() const { return impl_->isChecked; }

void Checkbox::SetChecked(bool checked) { impl_->isChecked = checked; }

Size Checkbox::CalcPreferredSize(const Theme& theme) const {
    /*    auto lineHeight = ImGui::GetTextLineHeight();
        auto height = lineHeight + 2.0 * ImGui::GetStyle().FramePadding.y;
        auto size = ImGui::GetFont()->CalcTextSizeA(theme.fontSize, 10000,
       10000, impl_->name.c_str());

        return Size(std::ceil(size.x + lineHeight), std::ceil(height));
     */
    auto em = ImGui::GetTextLineHeight();
    auto padding = ImGui::GetStyle().FramePadding;
    auto textSize = ImGui::GetFont()->CalcTextSizeA(theme.fontSize, 10000,
                                                    10000, impl_->name.c_str());
    int height = std::ceil(em + 2.0f * padding.y);
    auto checkboxWidth = height + padding.x;
    return Size(checkboxWidth + std::ceil(textSize.x + 2.0f * padding.x),
                height);
}

Widget::DrawResult Checkbox::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorPos(
            ImVec2(frame.x - context.uiOffsetX, frame.y - context.uiOffsetY));
    auto result = Widget::DrawResult::NONE;

    // ImGUI doesn't offer styling specific to checkboxes other than the
    // color of the checkmark, so we need to adjust the colors ourselves.
    if (impl_->isChecked) {
        ImGui::PushStyleColor(
                ImGuiCol_FrameBg,
                colorToImgui(context.theme.checkboxBackgroundOnColor));
        ImGui::PushStyleColor(
                ImGuiCol_FrameBgHovered,
                colorToImgui(context.theme.checkboxBackgroundHoverOnColor));
    } else {
        ImGui::PushStyleColor(
                ImGuiCol_FrameBg,
                colorToImgui(context.theme.checkboxBackgroundOffColor));
        ImGui::PushStyleColor(
                ImGuiCol_FrameBgHovered,
                colorToImgui(context.theme.checkboxBackgroundHoverOffColor));
    }

    ImGui::PushItemWidth(GetFrame().width);
    if (ImGui::Checkbox(impl_->name.c_str(), &impl_->isChecked)) {
        if (OnChecked) {
            OnChecked(impl_->isChecked);
        }
        result = Widget::DrawResult::CLICKED;
    }
    ImGui::PopItemWidth();

    ImGui::PopStyleColor(2);

    return result;
}

}  // namespace gui
}  // namespace open3d
