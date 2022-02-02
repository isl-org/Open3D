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
