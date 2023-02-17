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

#include "open3d/visualization/gui/VectorEdit.h"

#include <imgui.h>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
static int g_next_vector_edit_id = 1;
}

struct VectorEdit::Impl {
    std::string id_;
    Eigen::Vector3f value_;
    bool is_unit_vector_ = false;
    std::function<void(const Eigen::Vector3f&)> on_changed_;
};

VectorEdit::VectorEdit() : impl_(new VectorEdit::Impl()) {
    impl_->id_ = "##vectoredit_" + std::to_string(g_next_vector_edit_id++);
}

VectorEdit::~VectorEdit() {}

Eigen::Vector3f VectorEdit::GetValue() const { return impl_->value_; }

void VectorEdit::SetValue(const Eigen::Vector3f& val) {
    if (impl_->is_unit_vector_) {
        impl_->value_ = val.normalized();
    } else {
        impl_->value_ = val;
    }
}

void VectorEdit::SetOnValueChanged(
        std::function<void(const Eigen::Vector3f&)> on_changed) {
    impl_->on_changed_ = on_changed;
}

Size VectorEdit::CalcPreferredSize(const LayoutContext& context,
                                   const Constraints& constraints) const {
    auto em = std::ceil(ImGui::GetTextLineHeight());
    auto padding = ImGui::GetStyle().FramePadding;
    return Size(Widget::DIM_GROW, int(std::ceil(em + 2.0f * padding.y)));
}

Widget::DrawResult VectorEdit::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorScreenPos(
            ImVec2(float(frame.x), float(frame.y) - ImGui::GetScrollY()));

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,
                        0.0);  // macOS doesn't round text editing

    ImGui::PushStyleColor(
            ImGuiCol_FrameBg,
            colorToImgui(context.theme.text_edit_background_color));
    ImGui::PushStyleColor(
            ImGuiCol_FrameBgHovered,
            colorToImgui(context.theme.text_edit_background_color));
    ImGui::PushStyleColor(
            ImGuiCol_FrameBgActive,
            colorToImgui(context.theme.text_edit_background_color));

    auto result = Widget::DrawResult::NONE;
    DrawImGuiPushEnabledState();
    ImGui::PushItemWidth(float(GetFrame().width));
    if (ImGui::InputFloat3(impl_->id_.c_str(), impl_->value_.data())) {
        result = Widget::DrawResult::REDRAW;
    }
    ImGui::PopItemWidth();
    DrawImGuiPopEnabledState();
    DrawImGuiTooltip();

    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar();

    if (ImGui::IsItemDeactivatedAfterEdit()) {
        if (impl_->on_changed_) {
            impl_->on_changed_(impl_->value_);
        }
        result = Widget::DrawResult::REDRAW;
    }

    return result;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
