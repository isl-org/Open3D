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

#include "WidgetProxy.h"

#include "open3d/visualization/gui/Color.h"
#include "open3d/visualization/gui/Events.h"

namespace open3d {
namespace visualization {
namespace gui {
struct WidgetProxy::Impl {
    std::shared_ptr<Widget> widget_;
    bool need_layout_ = false;
};

WidgetProxy::WidgetProxy() : impl_(new WidgetProxy::Impl()) {}
WidgetProxy::~WidgetProxy() {}

std::shared_ptr<Widget> WidgetProxy::GetActiveWidget() const {
    return impl_->widget_;
}
void WidgetProxy::SetWidget(std::shared_ptr<Widget> widget) {
    impl_->widget_ = widget;
    impl_->need_layout_ = true;
}
std::shared_ptr<Widget> WidgetProxy::GetWidget() { return GetActiveWidget(); }
void WidgetProxy::AddChild(std::shared_ptr<Widget> child) {
    auto widget = GetActiveWidget();
    if (widget) {
        widget->AddChild(child);
    }
}

const std::vector<std::shared_ptr<Widget>> WidgetProxy::GetChildren() const {
    auto widget = GetActiveWidget();
    if (widget) {
        return widget->GetChildren();
    }
    return Widget::GetChildren();
}

const Rect& WidgetProxy::GetFrame() const {
    auto widget = GetActiveWidget();
    if (widget) {
        return widget->GetFrame();
    }
    return Widget::GetFrame();
}

void WidgetProxy::SetFrame(const Rect& f) {
    auto widget = GetActiveWidget();
    if (widget) {
        widget->SetFrame(f);
    }
    Widget::SetFrame(f);
}

const Color& WidgetProxy::GetBackgroundColor() const {
    auto widget = GetActiveWidget();
    if (widget) {
        return widget->GetBackgroundColor();
    }
    return Widget::GetBackgroundColor();
}

bool WidgetProxy::IsDefaultBackgroundColor() const {
    auto widget = GetActiveWidget();
    if (widget) {
        return widget->IsDefaultBackgroundColor();
    }
    return Widget::IsDefaultBackgroundColor();
}

void WidgetProxy::SetBackgroundColor(const Color& color) {
    auto widget = GetActiveWidget();
    if (widget) {
        widget->SetBackgroundColor(color);
    }
    Widget::SetBackgroundColor(color);
}

bool WidgetProxy::IsVisible() const {
    auto widget = GetActiveWidget();
    if (widget) {
        return Widget::IsVisible() && widget->IsVisible();
    }
    return false;
}

void WidgetProxy::SetVisible(bool vis) {
    auto widget = GetActiveWidget();
    if (widget) {
        widget->SetVisible(vis);
    }
}

bool WidgetProxy::IsEnabled() const {
    auto widget = GetActiveWidget();
    if (widget) {
        return Widget::IsEnabled() && widget->IsEnabled();
    }
    return false;
}

void WidgetProxy::SetEnabled(bool enabled) {
    auto widget = GetActiveWidget();
    if (widget) {
        widget->SetEnabled(enabled);
    }
}

void WidgetProxy::SetTooltip(const char* text) {
    auto widget = GetActiveWidget();
    if (widget) {
        widget->SetTooltip(text);
    }
    Widget::SetTooltip(text);
}

const char* WidgetProxy::GetTooltip() const {
    auto widget = GetActiveWidget();
    if (widget) {
        return widget->GetTooltip();
    }
    return Widget::GetTooltip();
}

Size WidgetProxy::CalcPreferredSize(const LayoutContext& context,
                                    const Constraints& constraints) const {
    auto widget = GetActiveWidget();
    if (widget) {
        return widget->CalcPreferredSize(context, constraints);
    }
    return Widget::CalcPreferredSize(context, constraints);
}

Size WidgetProxy::CalcMinimumSize(const LayoutContext& context) const {
    auto widget = GetActiveWidget();
    if (widget) {
        return widget->CalcMinimumSize(context);
    }
    return Widget::CalcMinimumSize(context);
}

void WidgetProxy::Layout(const LayoutContext& context) {
    auto widget = GetActiveWidget();
    if (widget) {
        widget->Layout(context);
    }
}

Widget::DrawResult WidgetProxy::Draw(const DrawContext& context) {
    if (!IsVisible()) {
        return DrawResult::NONE;
    }

    DrawResult result = DrawResult::NONE;
    auto widget = GetActiveWidget();
    if (widget) {
        result = widget->Draw(context);
    }
    if (impl_->need_layout_) {
        impl_->need_layout_ = false;
        result = DrawResult::RELAYOUT;
    }
    return result;
}

Widget::EventResult WidgetProxy::Mouse(const MouseEvent& e) {
    if (!IsVisible()) {
        return EventResult::IGNORED;
    }
    auto widget = GetActiveWidget();
    if (widget) {
        return widget->Mouse(e);
    }
    return EventResult::DISCARD;
}

Widget::EventResult WidgetProxy::Key(const KeyEvent& e) {
    if (!IsVisible()) {
        return EventResult::IGNORED;
    }
    auto widget = GetActiveWidget();
    if (widget) {
        return widget->Key(e);
    }
    return EventResult::DISCARD;
}

Widget::DrawResult WidgetProxy::Tick(const TickEvent& e) {
    auto result = DrawResult::NONE;
    auto widget = GetActiveWidget();
    if (widget) {
        result = widget->Tick(e);
    }
    return result;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
