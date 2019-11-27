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

#include "Layout.h"

#include <algorithm>

namespace open3d {
namespace gui {

namespace {
std::vector<std::vector<std::shared_ptr<Widget>>> calcColumns(int nCols, const std::vector<std::shared_ptr<Widget>>& children ) {
    std::vector<std::vector<std::shared_ptr<Widget>>> columns(nCols);
    int col = 0;
    for (auto &child : children) {
        columns[col++].push_back(child);
        if (col >= nCols) {
            col = 0;
        }
    }
    return columns;
}

std::vector<Size> calcColumnSizes(const std::vector<std::vector<std::shared_ptr<Widget>>>& columns,
                                  const Theme& theme) {
    std::vector<Size> sizes;
    sizes.reserve(columns.size());

    for (auto &col : columns) {
        int w = 0, h = 0;
        for (auto &widget : col) {
            auto preferred = widget->CalcPreferredSize(theme);
            w = std::max(w, preferred.width);
            h += preferred.height;
        }
        sizes.push_back(Size(w, h));
    }

    return sizes;
}

}

// ----------------------------------------------------------------------------
Margins::Margins()
    : left(0), top(0), right(0), bottom(0) {}
Margins::Margins(int px)
    : left(px), top(px), right(px), bottom(px) {}
Margins::Margins(int horizPx, int vertPx)
    : left(horizPx), top(vertPx), right(horizPx), bottom(vertPx) {}
Margins::Margins(int leftPx, int topPx, int rightPx, int bottomPx)
    : left(leftPx), top(topPx), right(rightPx), bottom(bottomPx) {}

// ----------------------------------------------------------------------------
struct VGrid::Impl {
    int nCols;
    int spacing;
    Margins margins;
};

VGrid::VGrid(int nCols, int spacing /*= 0*/,
             const Margins& margins /*= Margins()*/)
: impl_(new VGrid::Impl()) {
    impl_->nCols = nCols;
    impl_->spacing = spacing;
    impl_->margins = margins;
}

VGrid::~VGrid() {
}

Size VGrid::CalcPreferredSize(const Theme& theme) const {
    auto columns = calcColumns(impl_->nCols, GetChildren());
    auto columnSizes = calcColumnSizes(columns, theme);

    int width = 0, height = 0;
    for (auto &sz : columnSizes) {
        width += sz.width + impl_->spacing;
        height = std::max(height, sz.height) + impl_->spacing;
    }
    width -= impl_->spacing;  // remove "spacing" past the end
    height -= impl_->spacing;
    width = std::max(width, 0);  // in case width or height has no items
    height = std::max(height, 0);

    return Size(width + impl_->margins.left + impl_->margins.right,
                height + impl_->margins.top + impl_->margins.bottom);
}

void VGrid::Layout(const Theme& theme) {
    auto columns = calcColumns(impl_->nCols, GetChildren());
    auto columnSizes = calcColumnSizes(columns, theme);

    int x = GetFrame().GetLeft() + impl_->margins.left;
    for (size_t i = 0;  i < columns.size();  ++i) {
        int y = GetFrame().GetTop() + impl_->margins.top;
        for (auto &w : columns[i]) {
            auto preferred = w->CalcPreferredSize(theme);
            w->SetFrame(Rect(x, y, columnSizes[i].width, preferred.height));
            y += preferred.height + impl_->spacing;
        }
        x += columnSizes[i].width + impl_->spacing;
    }

    Super::Layout(theme);
}

} // gui
} // open3d
