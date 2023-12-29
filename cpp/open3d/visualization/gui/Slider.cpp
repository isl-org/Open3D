// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/Slider.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>

#include "open3d/visualization/gui/Theme.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
static int g_next_slider_id = 1;
}

struct Slider::Impl {
    Slider::Type type_;
    std::string id_;
    // A double has 53-bits of integer value, which should be enough for
    // anything we want a slider for. A slider isn't really useful for
    // a range of 2^53 anyway.
    double value_ = 0.0;
    double min_value_ = -1e35;
    double max_value_ = 1e35;
    std::function<void(double)> on_value_changed_;
};

Slider::Slider(Type type) : impl_(new Slider::Impl()) {
    impl_->id_ = "##slider_" + std::to_string(g_next_slider_id++);
    impl_->type_ = type;
}

Slider::~Slider() {}

int Slider::GetIntValue() const { return int(impl_->value_); }

double Slider::GetDoubleValue() const { return impl_->value_; }

void Slider::SetValue(double val) {
    impl_->value_ =
            std::max(impl_->min_value_, std::min(impl_->max_value_, val));
    if (impl_->type_ == INT) {
        impl_->value_ = std::round(impl_->value_);
    }
}

double Slider::GetMinimumValue() const { return impl_->min_value_; }

double Slider::GetMaximumValue() const { return impl_->max_value_; }

void Slider::SetLimits(double min_value, double max_value) {
    impl_->min_value_ = min_value;
    impl_->max_value_ = max_value;
    if (impl_->type_ == INT) {
        impl_->min_value_ = std::round(impl_->min_value_);
        impl_->max_value_ = std::round(impl_->max_value_);
    }
    SetValue(impl_->value_);  // make sure value is within new limits
}

void Slider::SetOnValueChanged(std::function<void(double)> on_value_changed) {
    impl_->on_value_changed_ = on_value_changed;
}

Size Slider::CalcPreferredSize(const LayoutContext& context,
                               const Constraints& constraints) const {
    auto line_height = ImGui::GetTextLineHeight();
    auto height = line_height + 2.0 * ImGui::GetStyle().FramePadding.y;

    return Size(Widget::DIM_GROW, int(std::ceil(height)));
}

Widget::DrawResult Slider::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorScreenPos(
            ImVec2(float(frame.x), float(frame.y) - ImGui::GetScrollY()));

    auto new_value = impl_->value_;
    DrawImGuiPushEnabledState();
    ImGui::PushItemWidth(float(GetFrame().width));
    if (impl_->type_ == INT) {
        int i_new_value = int(new_value);
        ImGui::SliderInt(impl_->id_.c_str(), &i_new_value,
                         int(impl_->min_value_), int(impl_->max_value_));
        new_value = double(i_new_value);
    } else {
        float f_new_value = float(new_value);
        ImGui::SliderFloat(impl_->id_.c_str(), &f_new_value,
                           float(impl_->min_value_), float(impl_->max_value_));
        new_value = double(f_new_value);
    }
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
