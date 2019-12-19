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

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "Gui.h"
#include "Events.h"
#include "Menu.h"

#include "Open3D/Visualization/Rendering/AbstractRenderInterface.h"

namespace open3d {
namespace gui {

class Menu;
class Renderer;
struct Theme;
class Widget;

class Window {
    friend class Application;
    friend class Renderer;
public:
    Window(const std::string& title, int width, int height);
    Window(const std::string& title, int x, int y, int width, int height);
    virtual ~Window();

    uint32_t GetID() const;

    const Theme& GetTheme() const;
    visualization::AbstractRenderInterface& GetRenderer() const;

    Size GetSize() const; // total interior size of window, including menubar
    Rect GetContentRect() const; // size available to widgets
    float GetScaling() const;
    Point GlobalToWindowCoord(int globalX, int globalY);

    bool IsVisible() const;
    void Show(bool vis = true);
    void Close();  // same as calling Application::RemoveWindow()

    std::shared_ptr<Menu> GetMenubar() const;
    void SetMenubar(std::shared_ptr<Menu> menu);

    void AddChild(std::shared_ptr<Widget> w);

    std::function<void(Menu::ItemId)> OnMenuItemSelected;

protected:
    virtual void Layout(const Theme& theme);

private:
    enum DrawResult { NONE, REDRAW };
    DrawResult OnDraw(float dtSec);
    DrawResult DrawOnce(float dtSec);
    void OnResize();
    void OnMouseEvent(const MouseEvent& e);
    void OnKeyEvent(const KeyEvent& e);
    void OnTextInput(const TextInputEvent& e);
    void* GetNativeDrawable() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}
