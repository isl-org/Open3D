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

#include "open3d/visualization/gui/StackedWidget.h"

#include <algorithm>  // for std::max, std::min

namespace open3d {
namespace visualization {
namespace gui {

struct StackedWidget::Impl {
    int selected_index_ = 0;
};

StackedWidget::StackedWidget() : impl_(new StackedWidget::Impl()) {}

StackedWidget::~StackedWidget() {}

void StackedWidget::SetSelectedIndex(int index) {
    impl_->selected_index_ = index;
}

int StackedWidget::GetSelectedIndex() const { return impl_->selected_index_; }

Size StackedWidget::CalcPreferredSize(const LayoutContext& context,
                                      const Constraints& constraints) const {
    Size size(0, 0);
    for (auto child : GetChildren()) {
        auto sz = child->CalcPreferredSize(context, constraints);
        size.width = std::max(size.width, sz.width);
        size.height = std::max(size.height, sz.height);
    }
    return size;
}

void StackedWidget::Layout(const LayoutContext& context) {
    auto& frame = GetFrame();
    for (auto child : GetChildren()) {
        child->SetFrame(frame);
    }

    Super::Layout(context);
}

Widget::DrawResult StackedWidget::Draw(const DrawContext& context) {
    // Don't Super, because Widget::Draw will draw all the children,
    // and we only want to draw the selected child.
    if (impl_->selected_index_ >= 0 &&
        impl_->selected_index_ < int(GetChildren().size())) {
        return GetChildren()[impl_->selected_index_]->Draw(context);
    }
    return DrawResult::NONE;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
