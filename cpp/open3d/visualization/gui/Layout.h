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

#include "open3d/visualization/gui/Widget.h"

namespace open3d {
namespace visualization {
namespace gui {

struct Margins {
    int left;
    int top;
    int right;
    int bottom;

    /// Margins are specified in pixels, which are not the same size on all
    /// monitors. It is best to use a multiple of
    /// Window::GetTheme().fontSize to specify margins. Theme::fontSize,
    /// represents 1em and is scaled according to the scaling factor of the
    /// window. For example, 0.5em (that is, 0.5 * theme.fontSize) is typically
    /// a good size for a margin.
    Margins();  // all values zero
    Margins(int px);
    Margins(int horiz_px, int vert_px);
    Margins(int left_px, int top_px, int right_px, int bottom_px);

    /// Convenience function that returns left + right
    int GetHoriz() const;
    /// Convenience function that returns top + bottom
    int GetVert() const;
};

/// Lays out widgets either horizontally or vertically.
/// Base class for Vert and Horiz.
class Layout1D : public Widget {
    using Super = Widget;

public:
    enum Dir { VERT, HORIZ };

    static void debug_PrintPreferredSizes(Layout1D* layout,
                                          const LayoutContext& context,
                                          const Constraints& constraints,
                                          int depth = 0);

    /// Spacing is in pixels; see the comment in Margin(). 1em is typically
    /// a good value for spacing.
    Layout1D(Dir dir,
             int spacing,
             const Margins& margins,
             const std::vector<std::shared_ptr<Widget>>& children);
    virtual ~Layout1D();

    int GetSpacing() const;
    const Margins& GetMargins() const;
    /// Sets spacing. Need to signal a relayout after calling (unless it is
    /// before a layout that will happen, such as before adding as a child).
    void SetSpacing(int spacing);
    /// Sets margins. Need to signal a relayout after calling (unless it is
    /// before a layout that will happen, such as before adding as a child).
    void SetMargins(const Margins& margins);

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;
    void Layout(const LayoutContext& context) override;

    /// Adds a fixed number of pixels after the previously added widget.
    void AddFixed(int size);
    /// Adds a virtual widget that takes up as much space as possible.
    /// This is useful for centering widgets: { stretch, w1, w2, stretch }
    /// or for aligning widgets to one side or the other:
    /// { stretch, ok, cancel }.
    void AddStretch();

public:
    class Fixed : public Widget {
    public:
        Fixed(int size, Dir dir);
        Size CalcPreferredSize(const LayoutContext& context,
                               const Constraints& constraints) const override;

    private:
        int size_;
        Dir dir_;
    };

    class Stretch : public Widget {
        Size CalcPreferredSize(const LayoutContext& context,
                               const Constraints& constraints) const override;
    };

protected:
    int GetMinorAxisPreferredSize() const;
    void SetMinorAxisPreferredSize(int size);

    Margins& GetMutableMargins();
    std::vector<std::shared_ptr<Widget>> GetVisibleChildren() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Lays out widgets vertically.
class Vert : public Layout1D {
public:
    static std::shared_ptr<Layout1D::Fixed> MakeFixed(int size);
    static std::shared_ptr<Layout1D::Stretch> MakeStretch();

    Vert();
    /// Spacing is in pixels; see the comment in Margin(). 1em is typically
    /// a good value for spacing.
    Vert(int spacing, const Margins& margins = Margins());
    Vert(int spacing,
         const Margins& margins,
         const std::vector<std::shared_ptr<Widget>>& children);
    virtual ~Vert();

    int GetPreferredWidth() const;
    void SetPreferredWidth(int w);
};

/// This is a vertical layout with a twisty + title that can be clicked on
/// to expand or collapse the layout. Collapsing the layout will hide all
/// the items and shrink the size of the layout to the height of the title.
class CollapsableVert : public Vert {
    using Super = Vert;

public:
    CollapsableVert(const char* text);
    CollapsableVert(const char* text,
                    int spacing,
                    const Margins& margins = Margins());
    virtual ~CollapsableVert();

    /// You will need to call Window::SetNeedsLayout() after this.
    /// (If you call this before the widnows is displayed everything
    /// will work out fine, as layout will automatically be called when
    /// the window is shown.)
    void SetIsOpen(bool is_open);

    /// Returns true if open and false if collapsed.
    bool GetIsOpen();

    FontId GetFontId() const;
    void SetFontId(FontId font_id);

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;
    void Layout(const LayoutContext& context) override;
    Widget::DrawResult Draw(const DrawContext& context) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// This a vertical layout that scrolls if it is smaller than its contents
class ScrollableVert : public Vert {
    using Super = Vert;

public:
    ScrollableVert();
    ScrollableVert(int spacing, const Margins& margins = Margins());
    ScrollableVert(int spacing,
                   const Margins& margins,
                   const std::vector<std::shared_ptr<Widget>>& children);
    virtual ~ScrollableVert();

    Widget::DrawResult Draw(const DrawContext& context) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Lays out widgets horizontally.
class Horiz : public Layout1D {
public:
    static std::shared_ptr<Layout1D::Fixed> MakeFixed(int size);
    static std::shared_ptr<Layout1D::Stretch> MakeStretch();
    static std::shared_ptr<Horiz> MakeCentered(std::shared_ptr<Widget> w);

    Horiz();
    /// Spacing is in pixels; see the comment in Margin(). 1em is typically
    /// a good value for spacing.
    Horiz(int spacing, const Margins& margins = Margins());
    Horiz(int spacing,
          const Margins& margins,
          const std::vector<std::shared_ptr<Widget>>& children);
    ~Horiz();

    int GetPreferredHeight() const;
    void SetPreferredHeight(int h);
};

/// Lays out widgets in a grid. The widgets are assigned to the next
/// horizontal column, and when all the columns in a row are used, a new
/// row will be created.
class VGrid : public Widget {
    using Super = Widget;

public:
    VGrid(int num_cols, int spacing = 0, const Margins& margins = Margins());
    virtual ~VGrid();

    int GetSpacing() const;
    const Margins& GetMargins() const;

    int GetPreferredWidth() const;
    void SetPreferredWidth(int w);

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;
    void Layout(const LayoutContext& context) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
