// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
