// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/WidgetStack.h"

#include <stack>

namespace open3d {
namespace visualization {
namespace gui {
struct WidgetStack::Impl {
    std::stack<std::shared_ptr<Widget>> widgets_;
    std::function<void(std::shared_ptr<Widget>)> on_top_callback_;
};

WidgetStack::WidgetStack() : impl_(new WidgetStack::Impl()) {}
WidgetStack::~WidgetStack() = default;

void WidgetStack::PushWidget(std::shared_ptr<Widget> widget) {
    impl_->widgets_.push(widget);
    SetWidget(widget);
}

std::shared_ptr<Widget> WidgetStack::PopWidget() {
    std::shared_ptr<Widget> ret;
    if (!impl_->widgets_.empty()) {
        ret = impl_->widgets_.top();
        impl_->widgets_.pop();
        if (!impl_->widgets_.empty()) {
            SetWidget(impl_->widgets_.top());
            if (impl_->on_top_callback_) {
                impl_->on_top_callback_(impl_->widgets_.top());
            }
        } else {
            SetWidget(nullptr);
        }
    }
    return ret;
}
void WidgetStack::SetOnTop(
        std::function<void(std::shared_ptr<Widget>)> onTopCallback) {
    impl_->on_top_callback_ = onTopCallback;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
