// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/ColorEdit.h"

#include <imgui.h>

#include <cmath>

#include "open3d/visualization/gui/Theme.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
static int g_next_color_edit_id = 1;
}

struct ColorEdit::Impl {
    std::string id_;
    Color value_;
    std::function<void(const Color&)> on_value_changed_;
};

ColorEdit::ColorEdit() : impl_(new ColorEdit::Impl()) {
    impl_->id_ = "##coloredit_" + std::to_string(g_next_color_edit_id++);
}

ColorEdit::~ColorEdit() {}

void ColorEdit::SetValue(const Color& color) { impl_->value_ = color; }

void ColorEdit::SetValue(const float r, const float g, const float b) {
    impl_->value_.SetColor(r, g, b);
}

const Color& ColorEdit::GetValue() const { return impl_->value_; }

void ColorEdit::SetOnValueChanged(
        std::function<void(const Color&)> on_value_changed) {
    impl_->on_value_changed_ = on_value_changed;
}

Size ColorEdit::CalcPreferredSize(const LayoutContext& context,
                                  const Constraints& constraints) const {
    auto line_height = ImGui::GetTextLineHeight();
    auto height = line_height + 2.0 * ImGui::GetStyle().FramePadding.y;

    return Size(Widget::DIM_GROW, int(std::ceil(height)));
}

ColorEdit::DrawResult ColorEdit::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorScreenPos(
            ImVec2(float(frame.x), float(frame.y) - ImGui::GetScrollY()));

    auto new_value = impl_->value_;
    DrawImGuiPushEnabledState();
    ImGui::PushItemWidth(float(GetFrame().width));
    ImGui::ColorEdit3(impl_->id_.c_str(), new_value.GetMutablePointer());
    ImGui::PopItemWidth();
    DrawImGuiPopEnabledState();
    DrawImGuiTooltip();

    if (impl_->value_ != new_value) {
        impl_->value_ = new_value;
        if (impl_->on_value_changed_) {
            impl_->on_value_changed_(new_value);
        }

        return Widget::DrawResult::REDRAW;
    }
    return Widget::DrawResult::NONE;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
