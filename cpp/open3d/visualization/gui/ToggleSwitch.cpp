// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/ToggleSwitch.h"

#include <imgui.h>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
float CalcSwitchWidth(float height) { return height * 1.55f; }

static int g_next_toggle_id = 1;
}  // namespace

struct ToggleSwitch::Impl {
    std::string name_;
    std::string id_;
    bool is_on_ = false;
    std::function<void(bool)> on_clicked_;
};

ToggleSwitch::ToggleSwitch(const char* name) : impl_(new ToggleSwitch::Impl()) {
    impl_->name_ = name;
    impl_->id_ =
            impl_->name_ + "##toggle_" + std::to_string(g_next_toggle_id++);
}

ToggleSwitch::~ToggleSwitch() {}

bool ToggleSwitch::GetIsOn() const { return impl_->is_on_; }

void ToggleSwitch::SetOn(bool checked) { impl_->is_on_ = checked; }

void ToggleSwitch::SetOnClicked(std::function<void(bool)> on_clicked) {
    impl_->on_clicked_ = on_clicked;
}

Size ToggleSwitch::CalcPreferredSize(const LayoutContext& context,
                                     const Constraints& constraints) const {
    auto em = ImGui::GetTextLineHeight();
    auto padding = ImGui::GetStyle().FramePadding;
    auto text_size = ImGui::GetFont()->CalcTextSizeA(
            float(context.theme.font_size), constraints.width, 10000,
            impl_->name_.c_str());
    int height = int(std::ceil(em + 2.0f * padding.y));
    auto switch_width = CalcSwitchWidth(height);
    return Size(int(switch_width + std::ceil(text_size.x + 2.0f * padding.x)),
                height);
}

Widget::DrawResult ToggleSwitch::Draw(const DrawContext& context) {
    auto& theme = context.theme;
    auto& frame = GetFrame();
    ImGui::SetCursorScreenPos(
            ImVec2(float(frame.x), float(frame.y) - ImGui::GetScrollY()));
    auto result = Widget::DrawResult::NONE;

    DrawImGuiPushEnabledState();
    ImGui::PushItemWidth(float(GetFrame().width));

    bool changed = false;
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    float height = frame.height;
    float width = CalcSwitchWidth(height);
    float radius = height * 0.50f;

    ImGui::InvisibleButton(impl_->name_.c_str(), ImVec2(width, height));
    DrawImGuiTooltip();  // button is separate obj, so needs its own call
    ImU32 track_color;
    ImU32 thumb_color = colorToImguiRGBA(theme.toggle_thumb_color);
    if (impl_->is_on_) {
        if (ImGui::IsItemHovered()) {
            track_color =
                    colorToImguiRGBA(theme.toggle_background_hover_on_color);
        } else {
            track_color = colorToImguiRGBA(theme.toggle_background_on_color);
        }
    } else {
        if (ImGui::IsItemHovered()) {
            track_color =
                    colorToImguiRGBA(theme.toggle_background_hover_off_color);
        } else {
            track_color = colorToImguiRGBA(theme.toggle_background_off_color);
        }
    }

    if (ImGui::IsItemClicked()) {
        impl_->is_on_ = !impl_->is_on_;
        changed = true;
    }
    draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), track_color,
                             radius);
    draw_list->AddRect(p, ImVec2(p.x + width, p.y + height),
                       colorToImguiRGBA(theme.border_color), radius);
    float thumb_offset =
            (impl_->is_on_ ? 1.0f : 0.0f) * (width - 2.0f * radius);
    draw_list->AddCircleFilled(
            ImVec2(p.x + radius + thumb_offset, p.y + radius), radius - 1.5f,
            thumb_color);

    if (changed) {
        if (impl_->on_clicked_) {
            impl_->on_clicked_(impl_->is_on_);
        }
        result = Widget::DrawResult::REDRAW;
    }

    ImGui::SameLine();
    ImGui::TextUnformatted(impl_->name_.c_str());
    ImGui::PopItemWidth();
    DrawImGuiPopEnabledState();
    DrawImGuiTooltip();

    return result;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
