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

#include "NumberEdit.h"

#include "Theme.h"
#include "Util.h"

#include <imgui.h>

#include <algorithm>  // for min, max
#include <cmath>
#include <sstream>
#include <unordered_set>

#include <string.h>

namespace open3d {
namespace gui {

namespace {
static int gNextNumberEditId = 1;
}

struct NumberEdit::Impl {
    std::string id;
    NumberEdit::Type type;
    // Double has 53-bits of integer range, which is sufficient for the
    // numbers that are reasonable for users to be entering. (Since JavaScript
    // only uses doubles, apparently it works for a lot more situations, too.)
    double value;
    double minValue = -2e9;  // -2 billion, which is roughly -INT_MAX
    double maxValue = 2e9;
    int nDecimalDigits = -1;
    std::function<void(double)> onChanged;
};

NumberEdit::NumberEdit(Type type) : impl_(new NumberEdit::Impl()) {
    impl_->type = type;
    std::stringstream s;
    s << "##numedit" << gNextNumberEditId++;
    impl_->id = s.str();
}

NumberEdit::~NumberEdit() {}

int NumberEdit::GetIntValue() const { return int(impl_->value); }

double NumberEdit::GetDoubleValue() const { return impl_->value; }

void NumberEdit::SetValue(double val) {
    if (impl_->type == INT) {
        impl_->value = std::round(val);
    } else {
        impl_->value = val;
    }
}

double NumberEdit::GetMinimumValue() const { return impl_->minValue; }

double NumberEdit::GetMaximumValue() const { return impl_->maxValue; }

void NumberEdit::SetLimits(double minValue, double maxValue) {
    if (impl_->type == INT) {
        impl_->minValue = std::round(minValue);
        impl_->maxValue = std::round(maxValue);
    } else {
        impl_->minValue = minValue;
        impl_->maxValue = maxValue;
    }
    impl_->value = std::min(maxValue, std::max(minValue, impl_->value));
}

void NumberEdit::SetDecimalPrecision(int nDigits) {
    impl_->nDecimalDigits = nDigits;
}

void NumberEdit::SetOnValueChanged(std::function<void(double)> onChanged) {
    impl_->onChanged = onChanged;
}

Size NumberEdit::CalcPreferredSize(const Theme &theme) const {
    int nMinDigits = std::ceil(std::log10(std::abs(impl_->minValue)));
    int nMaxDigits = std::ceil(std::log10(std::abs(impl_->maxValue)));
    int nDigits = std::max(6, std::max(nMinDigits, nMaxDigits));
    if (impl_->minValue < 0) {
        nDigits += 1;
    }

    auto pref = Super::CalcPreferredSize(theme);
    auto padding = pref.height - theme.fontSize;
    return Size((nDigits * theme.fontSize) / 2 + padding, pref.height);
}

Widget::DrawResult NumberEdit::Draw(const DrawContext &context) {
    auto &frame = GetFrame();
    ImGui::SetCursorPos(
            ImVec2(frame.x - context.uiOffsetX, frame.y - context.uiOffsetY));

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding,
                        0.0);  // macOS doesn't round text edit borders

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
    if (impl_->type == INT) {
        int iValue = impl_->value;
        if (ImGui::InputInt(impl_->id.c_str(), &iValue)) {
            SetValue(iValue);
            result = Widget::DrawResult::REDRAW;
        }
    } else {
        std::string fmt;
        if (impl_->nDecimalDigits >= 0) {
            char buff[32];
            snprintf(buff, 31, "%%.%df", impl_->nDecimalDigits);
            fmt = buff;
        } else {
            if (impl_->value < 10) {
                fmt = "%.3f";
            } else if (impl_->value < 100) {
                fmt = "%.2f";
            } else if (impl_->value < 1000) {
                fmt = "%.1f";
            } else {
                fmt = "%.0f";
            }
        }
        double dValue = impl_->value;
        if (ImGui::InputDouble(impl_->id.c_str(), &dValue, 0.0, 0.0,
                               fmt.c_str())) {
            SetValue(dValue);
            result = Widget::DrawResult::REDRAW;
        }
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
