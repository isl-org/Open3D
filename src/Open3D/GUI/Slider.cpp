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

#include "Slider.h"

#include "Theme.h"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace open3d {
namespace gui {

namespace {
static int gNextSliderId = 1;
}

struct Slider::Impl {
    Slider::Type type;
    std::string id;
    // A double has 24-bits of integer value, which should be enough for
    // anything we want a slider for. A slider isn't really useful for
    // a range of 2^24 = 16 million anyway.
    double value = 0.0;
    double minValue = -1e35;
    double maxValue = 1e35;
};

Slider::Slider(Type type) : impl_(new Slider::Impl()) {
    std::stringstream s;
    s << "##slider_" << gNextSliderId++;
    impl_->id = s.str();
    impl_->type = type;
}

Slider::~Slider() {}

int Slider::GetIntValue() const { return int(impl_->value); }

double Slider::GetDoubleValue() const { return impl_->value; }

void Slider::SetValue(double val) {
    impl_->value = std::max(impl_->minValue, std::min(impl_->maxValue, val));
    if (impl_->type == INT) {
        impl_->value = std::round(impl_->value);
    }
}

double Slider::GetMinimumValue() const { return impl_->minValue; }

double Slider::GetMaximumValue() const { return impl_->maxValue; }

void Slider::SetLimits(double minValue, double maxValue) {
    impl_->minValue = minValue;
    impl_->maxValue = maxValue;
    if (impl_->type == INT) {
        impl_->minValue = std::round(impl_->minValue);
        impl_->maxValue = std::round(impl_->maxValue);
    }
    SetValue(impl_->value);  // make sure value is within new limits
}

Size Slider::CalcPreferredSize(const Theme& theme) const {
    auto lineHeight = ImGui::GetTextLineHeight();
    auto height = lineHeight + 2.0 * ImGui::GetStyle().FramePadding.y;

    return Size(Widget::DIM_GROW, std::ceil(height));
}

Widget::DrawResult Slider::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorPos(
            ImVec2(frame.x - context.uiOffsetX, frame.y - context.uiOffsetY));

    float newValue = impl_->value;
    ImGui::PushItemWidth(GetFrame().width);
    if (impl_->type == INT) {
        int iNewValue = newValue;
        ImGui::SliderInt(impl_->id.c_str(), &iNewValue, impl_->minValue,
                         impl_->maxValue);
        newValue = double(iNewValue);
    } else {
        ImGui::SliderFloat(impl_->id.c_str(), &newValue, impl_->minValue,
                           impl_->maxValue);
    }
    ImGui::PopItemWidth();

    if (impl_->value != newValue) {
        impl_->value = newValue;
        if (OnValueChanged) {
            OnValueChanged(newValue);
        }
        return Widget::DrawResult::CLICKED;
    }
    return Widget::DrawResult::NONE;
}

}  // namespace gui
}  // namespace open3d
