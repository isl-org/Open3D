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

#include <functional>
#include <memory>

#include "open3d/visualization/gui/WidgetProxy.h"

namespace open3d {
namespace visualization {
namespace gui {

/// \brief WidgetStack manages multiple widgets in a stack.
///
/// WidgetStack, like \ref WidgetProxy, delegates at most one widget. Unlike
/// \ref WidgetProxy, it maintains multiple widgets. It saves all widgets
/// pushed into by \ref SetWidget and always shows the top one. The
/// WidgetStack is a subclass of \ref WidgetProxy, in another word, the topmost
/// widget will delegate it self to WidgetStack. \ref PopWidget will remove
/// the topmost widget and callback set by \ref SetOnTop taking the new
/// topmost widget will be called. The WidgetStack disappears in GUI if there
/// is no widget in stack.
///
/// Due to content and layout changing of the new widget, after \ref SetWidget
/// or \ref PopWidget the relayout of Window might need be called.
class WidgetStack : public WidgetProxy {
public:
    WidgetStack();
    ~WidgetStack() override;

    /// \brief Push a widget into stack so the it be the topmost widget.
    ///
    /// After a widget is pushed into stack, it will be shown in the GUI.
    /// Due to content and layout changing of the new widget, the relayout
    /// of Window might need be called.
    /// \param widget Widget to push into stack to be topmost showing wiget.
    void PushWidget(std::shared_ptr<Widget> widget);

    /// \brief Pop the top most widget.
    ///
    /// Due to content and layout changing of the new widget, the relayout
    /// of Window might need be called.
    /// \param widget Widget to push into stack to be topmost showing wiget.
    /// \return widget is popped, or NULL if no widget in the stack.
    std::shared_ptr<Widget> PopWidget();

    /// \brief Setup a callback while a widget is popped out and a new widget
    /// becomes the topmost one.
    ///
    /// Parameter of callback will be the new topmost widget.
    /// \param onTopCallback Callback function taking the new topmost widget.
    void SetOnTop(std::function<void(std::shared_ptr<Widget>)> onTopCallback);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
