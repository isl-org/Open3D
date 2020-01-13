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

#include "Button.h"

#include "Theme.h"

#include <imgui.h>

#include <cmath>
#include <string>

namespace open3d {
namespace gui {

struct Button::Impl {
    std::string title;
};

Button::Button(const char *title)
: impl_(new Button::Impl()) {
    impl_->title = title;
}

Button::~Button() {
}

Size Button::CalcPreferredSize(const Theme& theme) const {
    auto font = ImGui::GetFont();
    auto em = std::ceil(ImGui::GetTextLineHeight());
    auto size = font->CalcTextSizeA(theme.fontSize, 10000, 10000, impl_->title.c_str());
    return Size(std::ceil(size.x) + 2.0 * em, 2 * em);
}

Widget::DrawResult Button::Draw(const DrawContext& context, const float) {
    auto &frame = GetFrame();
    ImGui::SetCursorPos(ImVec2(frame.x - context.uiOffsetX,
                               frame.y - context.uiOffsetY));
    if (ImGui::Button(impl_->title.c_str(), ImVec2(GetFrame().width, GetFrame().height))) {
        if (this->OnClicked) {
            this->OnClicked();
        }
        return Widget::DrawResult::CLICKED;
    } else {
        return Widget::DrawResult::NONE;
    }
}

} // gui
} // open3d
