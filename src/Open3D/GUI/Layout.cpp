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

std::vector<int> calcMajor(const Theme& theme, Layout1D::Dir dir,
                           const std::vector<std::shared_ptr<Widget>>& children,
                           int *minor = nullptr) {
    std::vector<Size> preferredSizes;
    preferredSizes.reserve(children.size());
    for (auto &child : children) {
        preferredSizes.push_back(child->CalcPreferredSize(theme));
    }

    // Preferred size in the minor direction is the maximum preferred size,
    // not including the items that want to be as big as possible (unless they
    // are the only items).
    int other = 0;
    int nOtherMaxgrowItems = 0;
    std::vector<int> major;
    major.reserve(preferredSizes.size());
    if (dir == Layout1D::VERT) {
        for (auto &preferred : preferredSizes) {
            major.push_back(preferred.height);
            if (preferred.width >= Widget::DIM_GROW) {
                nOtherMaxgrowItems += 1;
            } else {
                other = std::max(other, preferred.width);
            }
        }
    } else {
        for (auto &preferred : preferredSizes) {
            major.push_back(preferred.width);
            if (preferred.height >= Widget::DIM_GROW) {
                nOtherMaxgrowItems += 1;
            } else {
                other = std::max(other, preferred.height);
            }
        }
    }

    if (other == 0 && nOtherMaxgrowItems > 0) {
        other = Widget::DIM_GROW;
    }

    if (minor) {
        *minor = other;
    }
    return major;
}

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

int Margins::GetHoriz() const {
    return this->left + this->right;
}

int Margins::GetVert() const {
    return this->top + this->bottom;
}

// ----------------------------------------------------------------------------
struct Layout1D::Impl {
    Layout1D::Dir dir;
    int spacing;
    Margins margins;
};

Layout1D::Fixed::Fixed(int size) : size_(size) {
}

Size Layout1D::Fixed::CalcPreferredSize(const Theme& theme) const {
    return Size(size_, size_);
}

Size Layout1D::Stretch::CalcPreferredSize(const Theme& theme) const {
    return Size(0, 0);
}

Layout1D::Layout1D(Dir dir, int spacing, const Margins& margins,
                   const std::vector<std::shared_ptr<Widget>>& children)
: Super(children), impl_(new Layout1D::Impl()) {
    impl_->dir = dir;
    impl_->spacing = spacing;
    impl_->margins = margins;
}

Layout1D::~Layout1D() {
}

Size Layout1D::CalcPreferredSize(const Theme& theme) const {
    int minor;
    std::vector<int> major = calcMajor(theme, impl_->dir, GetChildren(),
                                       &minor);

    int totalSpacing = impl_->spacing * (major.size() - 1);
    int majorSize = 0;
    for (auto &size : major) {
        majorSize += size;
    }

    if (impl_->dir == VERT) {
        return Size(minor + impl_->margins.GetHoriz(),
                    majorSize + impl_->margins.GetVert() + totalSpacing);
    } else {
        return Size(majorSize + impl_->margins.GetHoriz() + totalSpacing,
                    minor + impl_->margins.GetVert());
    }
}

void Layout1D::Layout(const Theme& theme) {
    auto frame = GetFrame();
    auto &children = GetChildren();
    std::vector<int> major = calcMajor(theme, impl_->dir, children, nullptr);
    int total = 0, nStretch = 0, nGrow = 0;
    for (auto &mj : major) {
        total += mj;
        if (mj <= 0) {
            nStretch += 1;
        }
        if (mj >= Widget::DIM_GROW) {
            nGrow += 1;
        }
    }
    int frameSize = (impl_->dir == VERT ? frame.height : frame.width);
    auto totalExtra = frameSize - total - impl_->spacing * (major.size() - 1);
    if (nStretch > 0 && frameSize > total) {
        auto stretch = totalExtra / nStretch;
        auto leftoverStretch = totalExtra - stretch * nStretch;
        for (size_t i = 0;  i < major.size();  ++i) {
            if (major[i] <= 0) {
                major[i] = stretch;
                if (leftoverStretch > 0) {
                    major[i] += 1;
                    leftoverStretch -= 1;
                }
            }
        }
    } else if (nGrow > 0 && frameSize < total) {
        auto totalExcess = total - (frameSize - impl_->margins.GetVert() -
                                    impl_->spacing * (major.size() - 1));
        auto excess = totalExcess / nGrow;
        auto leftover = totalExcess - excess * nStretch;
        for (size_t i = 0;  i < major.size();  ++i) {
            if (major[i] >= Widget::DIM_GROW) {
                major[i] -= excess;
                if (leftover > 0) {
                    major[i] -= 1;
                    leftover -= 1;
                }
            }
        }
    } /*else if (totalExtra != 0) {
        int each = totalExtra / major.size();
        int leftover = totalExtra - major.size() * each;
        for (size_t i = 0;  i < major.size();  ++i) {
            major[i] += each;
            if (leftover != 0) {
                auto one = leftover / std::abs(leftover); // 1 with leftovers' sign
                major[i] += one;
                leftover -= one;
            }
        }
    } */

    int x = frame.GetLeft() + impl_->margins.left;
    int y = frame.GetTop() + impl_->margins.top;
    if (impl_->dir == VERT) {
        int minor = frame.width - impl_->margins.GetHoriz();
        for (size_t i = 0;  i < children.size();  ++i) {
            children[i]->SetFrame(Rect(x, y, minor, major[i]));
            y += major[i] + impl_->spacing;
        }
    } else {
        int minor = frame.height - impl_->margins.GetVert();
        for (size_t i = 0;  i < children.size();  ++i) {
            children[i]->SetFrame(Rect(x, y, major[i], minor));
            x += major[i] + impl_->spacing;
        }
    }

    Super::Layout(theme);
}

// ----------------------------------------------------------------------------
std::shared_ptr<Layout1D::Fixed> Vert::MakeFixed(int size) {
    return std::make_shared<Layout1D::Fixed>(size);
}

std::shared_ptr<Layout1D::Stretch> Vert::MakeStretch() {
    return std::make_shared<Layout1D::Stretch>();
}

Vert::Vert()
: Layout1D(VERT, 0, Margins(), {}) {
}

Vert::Vert(int spacing /*= 0*/, const Margins& margins /*= Margins()*/)
: Layout1D(VERT, spacing, margins, {}) {
}

Vert::Vert(int spacing, const Margins& margins,
           const std::vector<std::shared_ptr<Widget>>& children)
: Layout1D(VERT, spacing, margins, children) {
}

Vert::~Vert() {
}

// ----------------------------------------------------------------------------
std::shared_ptr<Layout1D::Fixed> Horiz::MakeFixed(int size) {
    return std::make_shared<Layout1D::Fixed>(size);
}

std::shared_ptr<Layout1D::Stretch> Horiz::MakeStretch() {
    return std::make_shared<Layout1D::Stretch>();
}

std::shared_ptr<Horiz> Horiz::MakeCentered(std::shared_ptr<Widget> w) {
    return std::make_shared<Horiz>(0, Margins(),
            std::vector<std::shared_ptr<Widget>>({ Horiz::MakeStretch(),
                                                   w,
                                                   Horiz::MakeStretch() }));
}

Horiz::Horiz()
: Layout1D(HORIZ, 0, Margins(), {}) {
}

Horiz::Horiz(int spacing /*= 0*/, const Margins& margins /*= Margins()*/)
: Layout1D(HORIZ, spacing, margins, {}) {
}

Horiz::Horiz(int spacing, const Margins& margins,
           const std::vector<std::shared_ptr<Widget>>& children)
: Layout1D(HORIZ, spacing, margins, children) {
}

Horiz::~Horiz() {
}

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
