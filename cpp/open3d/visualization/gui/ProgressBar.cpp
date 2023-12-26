// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/ProgressBar.h"

#include <imgui.h>

#include <cmath>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

struct ProgressBar::Impl {
    float value_ = 0.0f;
};

ProgressBar::ProgressBar() : impl_(new ProgressBar::Impl()) {}

ProgressBar::~ProgressBar() {}

/// ProgressBar values ranges from 0.0 (incomplete) to 1.0 (complete)
void ProgressBar::SetValue(float value) { impl_->value_ = value; }

float ProgressBar::GetValue() const { return impl_->value_; }

Size ProgressBar::CalcPreferredSize(const LayoutContext& context,
                                    const Constraints& constraints) const {
    return Size(constraints.width,
                int(std::ceil(0.25 * context.theme.font_size)));
}

Widget::DrawResult ProgressBar::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    auto fg = context.theme.border_color;
    auto color = colorToImguiRGBA(fg);
    float rounding = frame.height / 2.0f;

    ImGui::GetWindowDrawList()->AddRect(
            ImVec2(float(frame.x), float(frame.y) - ImGui::GetScrollY()),
            ImVec2(float(frame.GetRight()),
                   float(frame.GetBottom()) - ImGui::GetScrollY()),
            color, rounding);
    float x = float(frame.x) + float(frame.width) * impl_->value_;
    x = std::max(x, float(frame.x + rounding));

    ImGui::GetWindowDrawList()->AddRectFilled(
            ImVec2(float(frame.x), float(frame.y) - ImGui::GetScrollY()),
            ImVec2(float(x), float(frame.GetBottom()) - ImGui::GetScrollY()),
            color, frame.height / 2.0f);
    return DrawResult::NONE;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
