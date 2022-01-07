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

#pragma once

#include <memory>
#include <vector>

#include "open3d/visualization/gui/Gui.h"

namespace open3d {

namespace visualization {
namespace rendering {
class Renderer;
}
}  // namespace visualization

namespace visualization {
namespace gui {

class Color;
struct MouseEvent;
struct KeyEvent;
struct TickEvent;
struct Theme;

struct LayoutContext {
    const Theme& theme;
    FontContext& fonts;
};

struct DrawContext {
    const Theme& theme;
    visualization::rendering::Renderer& renderer;
    FontContext& fonts;
    int uiOffsetX;
    int uiOffsetY;
    int screenWidth;
    int screenHeight;
    int emPx;
    float frameDelta;  // in seconds
};

class Widget {
    friend class Window;

public:
    Widget();
    explicit Widget(const std::vector<std::shared_ptr<Widget>>& children);
    virtual ~Widget();

    virtual void AddChild(std::shared_ptr<Widget> child);
    virtual const std::vector<std::shared_ptr<Widget>> GetChildren() const;

    /// Returns the frame size in pixels.
    virtual const Rect& GetFrame() const;
    /// The frame is in pixels. The size of a pixel varies on different
    /// and operatings sytems now frequently scale text sizes on high DPI
    /// monitors. Prefer using a Layout to using this function, but if you
    /// must use it, it is best to use a multiple of
    /// Window::GetTheme().fontSize, which represents 1em and is scaled
    /// according to the scaling factor of the window.
    virtual void SetFrame(const Rect& f);

    virtual const Color& GetBackgroundColor() const;
    virtual bool IsDefaultBackgroundColor() const;
    virtual void SetBackgroundColor(const Color& color);

    virtual bool IsVisible() const;
    virtual void SetVisible(bool vis);

    virtual bool IsEnabled() const;
    virtual void SetEnabled(bool enabled);

    virtual void SetTooltip(const char* text);
    virtual const char* GetTooltip() const;

    static constexpr int DIM_GROW = 10000;
    struct Constraints {
        int width = DIM_GROW;
        int height = DIM_GROW;
    };
    virtual Size CalcPreferredSize(const LayoutContext& context,
                                   const Constraints& constraints) const;

    virtual Size CalcMinimumSize(const LayoutContext& context) const;

    virtual void Layout(const LayoutContext& context);

    enum class DrawResult { NONE, REDRAW, RELAYOUT };
    /// Draws the widget. If this is a Dear ImGUI widget, this is where
    /// the actual event processing happens. Return NONE if no action
    /// needs to be taken, REDRAW if the widget needs to be redrawn
    /// (e.g. its value changed), and RELAYOUT if the widget wishes to
    /// change size.
    virtual DrawResult Draw(const DrawContext& context);

    enum class EventResult { IGNORED, CONSUMED, DISCARD };

    /// Widgets that use Dear ImGUI should not need to override this,
    /// as Dear ImGUI will take care of all the mouse handling during
    /// the Draw().
    virtual EventResult Mouse(const MouseEvent& e);

    /// Widgets that use Dear ImGUI should not need to override this,
    /// as Dear ImGUI will take care of all the key handling during
    /// the Draw().
    virtual EventResult Key(const KeyEvent& e);

    /// Tick events are sent regularly and allow for things like smoothly
    /// moving the camera based on keys that are pressed, or animations.
    /// Return DrawResult::REDRAW if you want to be redrawn.
    virtual DrawResult Tick(const TickEvent& e);

protected:
    void DrawImGuiPushEnabledState();
    void DrawImGuiPopEnabledState();
    void DrawImGuiTooltip();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
