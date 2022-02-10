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

#include "open3d/visualization/gui/NumberEdit.h"

#include <imgui.h>
#include <string.h>  // for snprintf

#include <algorithm>  // for min, max
#include <cmath>
#include <unordered_set>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {
static int g_next_number_edit_id = 1;
}

struct NumberEdit::Impl {
    std::string id_;
    NumberEdit::Type type_;
    // Double has 53-bits of integer range, which is sufficient for the
    // numbers that are reasonable for users to be entering. (Since JavaScript
    // only uses doubles, apparently it works for a lot more situations, too.)
    double value_;
    double min_value_ = -2e9;  // -2 billion, which is roughly -INT_MAX
    double max_value_ = 2e9;
    int num_decimal_digits_ = -1;
    int preferred_width_ = 0;
    std::function<void(double)> on_changed_;
};

NumberEdit::NumberEdit(Type type) : impl_(new NumberEdit::Impl()) {
    impl_->type_ = type;
    impl_->id_ = "##numedit" + std::to_string(g_next_number_edit_id++);
}

NumberEdit::~NumberEdit() {}

int NumberEdit::GetIntValue() const { return int(impl_->value_); }

double NumberEdit::GetDoubleValue() const { return impl_->value_; }

void NumberEdit::SetValue(double val) {
    if (impl_->type_ == INT) {
        impl_->value_ = std::round(val);
    } else {
        impl_->value_ = val;
    }
}

double NumberEdit::GetMinimumValue() const { return impl_->min_value_; }

double NumberEdit::GetMaximumValue() const { return impl_->max_value_; }

void NumberEdit::SetLimits(double min_value, double max_value) {
    if (impl_->type_ == INT) {
        impl_->min_value_ = std::round(min_value);
        impl_->max_value_ = std::round(max_value);
    } else {
        impl_->min_value_ = min_value;
        impl_->max_value_ = max_value;
    }
    impl_->value_ = std::min(max_value, std::max(min_value, impl_->value_));
}

int NumberEdit::GetDecimalPrecision() { return impl_->num_decimal_digits_; }

void NumberEdit::SetDecimalPrecision(int num_digits) {
    impl_->num_decimal_digits_ = num_digits;
}

void NumberEdit::SetPreferredWidth(int width) {
    impl_->preferred_width_ = width;
}

void NumberEdit::SetOnValueChanged(std::function<void(double)> on_changed) {
    impl_->on_changed_ = on_changed;
}

Size NumberEdit::CalcPreferredSize(const LayoutContext& context,
                                   const Constraints& constraints) const {
    int num_min_digits =
            int(std::ceil(std::log10(std::abs(impl_->min_value_))));
    int num_max_digits =
            int(std::ceil(std::log10(std::abs(impl_->max_value_))));
    int num_digits = std::max(6, std::max(num_min_digits, num_max_digits));
    if (impl_->min_value_ < 0) {
        num_digits += 1;
    }

    int height = int(std::round(ImGui::GetTextLineHeightWithSpacing()));
    auto padding = height - int(std::round(ImGui::GetTextLineHeight()));
    int incdec_width = 0;
    if (impl_->type_ == INT) {
        // padding is for the spacing between buttons and between text box
        incdec_width = 2 * height + padding;
    }

    int width =
            (num_digits * context.theme.font_size) / 2 + padding + incdec_width;
    if (impl_->preferred_width_ > 0) {
        width = impl_->preferred_width_;
    }
    return Size(width, height);
}

Widget::DrawResult NumberEdit::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorScreenPos(
            ImVec2(float(frame.x), float(frame.y) - ImGui::GetScrollY()));

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,
                        0.0);  // macOS doesn't round text edit borders

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
    if (impl_->type_ == INT) {
        int ivalue = int(impl_->value_);
        if (ImGui::InputInt(impl_->id_.c_str(), &ivalue)) {
            SetValue(ivalue);
            result = Widget::DrawResult::REDRAW;
        }
    } else {
        std::string fmt;
        if (impl_->num_decimal_digits_ >= 0) {
            char buff[32];
            snprintf(buff, 31, "%%.%df", impl_->num_decimal_digits_);
            fmt = buff;
        } else {
            if (impl_->value_ < 10) {
                fmt = "%.3f";
            } else if (impl_->value_ < 100) {
                fmt = "%.2f";
            } else if (impl_->value_ < 1000) {
                fmt = "%.1f";
            } else {
                fmt = "%.0f";
            }
        }
        double dvalue = impl_->value_;
        if (ImGui::InputDouble(impl_->id_.c_str(), &dvalue, 0.0, 0.0,
                               fmt.c_str())) {
            SetValue(dvalue);
            result = Widget::DrawResult::REDRAW;
        }
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
