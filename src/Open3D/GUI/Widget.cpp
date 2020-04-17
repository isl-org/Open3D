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

#include "Widget.h"

#include "Color.h"
#include "Events.h"

#include <imgui.h>
#include <imgui_internal.h>

namespace open3d {
namespace gui {

// This should not be Color(0, 0, 0, 0), since transparent is a valid and
// common background color to want.
static const Color DEFAULT_BGCOLOR(0.001, 0.001, 0.001, 0);

struct Widget::Impl {
    Rect frame;
    Color bgColor = DEFAULT_BGCOLOR;
    std::vector<std::shared_ptr<Widget>> children;
    bool isVisible = true;
    bool isEnabled = true;
    bool popDisabledFlagsAtEndOfDraw = false;
};

Widget::Widget() : impl_(new Widget::Impl()) {}

Widget::Widget(const std::vector<std::shared_ptr<Widget>>& children)
    : impl_(new Widget::Impl()) {
    impl_->children = children;
}

Widget::~Widget() {}

void Widget::AddChild(std::shared_ptr<Widget> child) {
    impl_->children.push_back(child);
}

const std::vector<std::shared_ptr<Widget>> Widget::GetChildren() const {
    return impl_->children;
}

const Rect& Widget::GetFrame() const { return impl_->frame; }

void Widget::SetFrame(const Rect& f) { impl_->frame = f; }

const Color& Widget::GetBackgroundColor() const { return impl_->bgColor; }

bool Widget::IsDefaultBackgroundColor() const {
    return (impl_->bgColor == DEFAULT_BGCOLOR);
}

void Widget::SetBackgroundColor(const Color& color) { impl_->bgColor = color; }

bool Widget::IsVisible() const { return impl_->isVisible; }

void Widget::SetVisible(bool vis) { impl_->isVisible = vis; }

bool Widget::IsEnabled() const { return impl_->isEnabled; }

void Widget::SetEnabled(bool enabled) { impl_->isEnabled = enabled; }

Size Widget::CalcPreferredSize(const Theme&) const {
    return Size(DIM_GROW, DIM_GROW);
}

void Widget::Layout(const Theme& theme) {
    for (auto& child : impl_->children) {
        child->Layout(theme);
    }
}

Widget::DrawResult Widget::Draw(const DrawContext& context) {
    if (!impl_->isVisible) {
        return DrawResult::NONE;
    }

    DrawResult result = DrawResult::NONE;
    for (auto& child : impl_->children) {
        if (child->IsVisible()) {
            auto r = child->Draw(context);
            // The mouse can only be over one item, so there should never
            // be multiple items returning non-NONE.
            if (r != DrawResult::NONE) {
                result = r;
            }
        }
    }
    return result;
}

void Widget::DrawImGuiPushEnabledState() {
    if (!IsEnabled()) {
        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha,
                            ImGui::GetStyle().Alpha * 0.5f);
    }
    // As an immediate mode GUI, responses to UI events can happen
    // during a draw. Store what the disabled flag was at the
    // beginning of the draw, so we know how many things to pop
    // to clean up. (An example of when this is needed is a reset
    // button: after clicking it, it will probably disable itself
    // since there is nothing to reset now)
    impl_->popDisabledFlagsAtEndOfDraw = !IsEnabled();
}

void Widget::DrawImGuiPopEnabledState() {
    if (impl_->popDisabledFlagsAtEndOfDraw) {
        ImGui::PopStyleVar();
        ImGui::PopItemFlag();
    }
}

Widget::EventResult Widget::Mouse(const MouseEvent& e) {
    if (!impl_->isVisible) {
        return EventResult::IGNORED;
    }

    // Iterate backwards so that we send mouse events from the top down.
    for (auto it = impl_->children.rbegin(); it != impl_->children.rend();
         ++it) {
        if ((*it)->GetFrame().Contains(e.x, e.y)) {
            auto result = (*it)->Mouse(e);
            if (result != EventResult::IGNORED) {
                return result;
            }
        }
    }

    // If we get here then this event is either for an ImGUI widget,
    // in which case we should not process this event further (ImGUI will
    // do it later), or this is an empty widget like a panel or something,
    // which eats events (it doesn't handle the event [e.g. button down]
    // and nor should anything else).
    return EventResult::DISCARD;
}

Widget::EventResult Widget::Key(const KeyEvent& e) {
    return EventResult::DISCARD;
}

Widget::DrawResult Widget::Tick(const TickEvent& e) {
    auto result = DrawResult::NONE;
    for (auto it = impl_->children.begin(); it != impl_->children.end(); ++it) {
        if ((*it)->Tick(e) == DrawResult::REDRAW) {
            result = DrawResult::REDRAW;
        }
    }
    return result;
}

}  // namespace gui
}  // namespace open3d
