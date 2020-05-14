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

#include "VectorEdit.h"

#include "Theme.h"
#include "Util.h"

#include <imgui.h>

#include <sstream>

namespace open3d {
namespace gui {

namespace {
static int gNextVectorEditId = 1;
}

struct VectorEdit::Impl {
    std::string id;
    Eigen::Vector3f value;
    bool isUnitVector = false;
    std::function<void(const Eigen::Vector3f&)> onChanged;
};

VectorEdit::VectorEdit() : impl_(new VectorEdit::Impl()) {
    std::stringstream s;
    s << "##vectoredit" << gNextVectorEditId++ << std::endl;
}

VectorEdit::~VectorEdit() {}

Eigen::Vector3f VectorEdit::GetValue() const { return impl_->value; }

void VectorEdit::SetValue(const Eigen::Vector3f& val) {
    if (impl_->isUnitVector) {
        impl_->value = val.normalized();
    } else {
        impl_->value = val;
    }
}

void VectorEdit::SetOnValueChanged(
        std::function<void(const Eigen::Vector3f&)> onChanged) {
    impl_->onChanged = onChanged;
}

Size VectorEdit::CalcPreferredSize(const Theme& theme) const {
    auto em = std::ceil(ImGui::GetTextLineHeight());
    auto padding = ImGui::GetStyle().FramePadding;
    return Size(Widget::DIM_GROW, std::ceil(em + 2.0f * padding.y));
}

Widget::DrawResult VectorEdit::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorPos(
            ImVec2(frame.x - context.uiOffsetX, frame.y - context.uiOffsetY));

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,
                        0.0);  // macOS doesn't round text editing

    ImGui::PushStyleColor(
            ImGuiCol_FrameBg,
            util::colorToImgui(context.theme.textEditBackgroundColor));
    ImGui::PushStyleColor(
            ImGuiCol_FrameBgHovered,
            util::colorToImgui(context.theme.textEditBackgroundColor));
    ImGui::PushStyleColor(
            ImGuiCol_FrameBgActive,
            util::colorToImgui(context.theme.textEditBackgroundColor));

    auto result = Widget::DrawResult::NONE;
    DrawImGuiPushEnabledState();
    ImGui::PushItemWidth(GetFrame().width);
    if (ImGui::InputFloat3(impl_->id.c_str(), impl_->value.data(), 3)) {
        result = Widget::DrawResult::REDRAW;
    }
    ImGui::PopItemWidth();
    DrawImGuiPopEnabledState();

    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar();

    if (ImGui::IsItemDeactivatedAfterEdit()) {
        if (impl_->onChanged) {
            impl_->onChanged(impl_->value);
        }
        result = Widget::DrawResult::REDRAW;
    }

    return result;
}

}  // namespace gui
}  // namespace open3d
