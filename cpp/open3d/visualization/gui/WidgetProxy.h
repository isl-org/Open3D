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
#include "open3d/visualization/gui/Widget.h"

namespace open3d {

namespace visualization {
namespace gui {

/// \class WidgetProxy
///
/// \brief Widget container to delegate any widget dynamically.
///
/// Widget can not be managed dynamically. Although it is allowed
/// to add more child widgets, it's impossible to replace some child
/// with new on or remove children. WidgetProxy is designed to solve
/// this problem.
///
/// When WidgetProxy is created, it's invisible and disabled, so it
/// won't be drawn or layout, seeming like it does not exist. When
/// a widget is set by \ref SetWidget, all \ref Widget's APIs will be
/// conducted to that child widget. It looks like WidgetProxy is
/// that widget.
///
/// At any time, a new widget could be set, to replace the old one.
/// and the old widget will be destroyed.
///
/// Due to the content changing after a new widget is set or cleared,
/// a relayout of Window might be called after SetWidget.
///
/// The delegated widget could be retrieved by \ref GetWidget in case you
/// need to access it directly, like get check status of a CheckBox.
///
/// API other than \ref SetWidget and \ref GetWidget has completely
/// same function as \ref Widget.
class WidgetProxy : public Widget {
public:
    WidgetProxy();
    ~WidgetProxy() override;

    void AddChild(std::shared_ptr<Widget> child) override;
    const std::vector<std::shared_ptr<Widget>> GetChildren() const override;

    /// \brief set a new widget to be delegated by this one.
    ///
    /// After SetWidget, the previously delegated widget will be abandon,
    /// all calls to \ref Widget's API will be conducted to \p widget.
    ///
    /// Before any SetWidget call, this widget is invisible and disabled,
    /// seems it does not exist because it won't be drawn or in a layout.
    ///
    /// \param widget Any widget to be delegated. Set to NULL to clear
    ///               current delegated proxy.
    virtual void SetWidget(std::shared_ptr<Widget> widget);

    /// \brief Retrieve current delegated widget.
    ///
    /// \return Instance of current delegated widget set by \ref SetWidget.
    ///         An empty pointer will be returned if there is none.
    virtual std::shared_ptr<Widget> GetWidget();

    const Rect& GetFrame() const override;
    void SetFrame(const Rect& f) override;

    const Color& GetBackgroundColor() const override;
    bool IsDefaultBackgroundColor() const override;
    void SetBackgroundColor(const Color& color) override;

    bool IsVisible() const override;
    void SetVisible(bool vis) override;

    bool IsEnabled() const override;
    void SetEnabled(bool enabled) override;
    void SetTooltip(const char* text) override;
    const char* GetTooltip() const override;
    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;
    Size CalcMinimumSize(const LayoutContext& context) const override;
    void Layout(const LayoutContext& context) override;
    DrawResult Draw(const DrawContext& context) override;
    EventResult Mouse(const MouseEvent& e) override;
    EventResult Key(const KeyEvent& e) override;
    DrawResult Tick(const TickEvent& e) override;

protected:
    virtual std::shared_ptr<Widget> GetActiveWidget() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
