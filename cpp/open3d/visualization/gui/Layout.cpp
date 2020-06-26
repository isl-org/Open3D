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

#include "open3d/visualization/gui/Layout.h"

#include <imgui.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {

std::vector<int> CalcMajor(const Theme& theme,
                           Layout1D::Dir dir,
                           const std::vector<std::shared_ptr<Widget>>& children,
                           int* minor = nullptr) {
    std::vector<Size> preferred_sizes;
    preferred_sizes.reserve(children.size());
    for (auto& child : children) {
        preferred_sizes.push_back(child->CalcPreferredSize(theme));
    }

    // Preferred size in the minor direction is the maximum preferred size,
    // not including the items that want to be as big as possible (unless they
    // are the only items).
    int other = 0;
    int num_other_maxgrow_items = 0;
    std::vector<int> major;
    major.reserve(preferred_sizes.size());
    if (dir == Layout1D::VERT) {
        for (auto& preferred : preferred_sizes) {
            major.push_back(preferred.height);
            if (preferred.width >= Widget::DIM_GROW) {
                num_other_maxgrow_items += 1;
            } else {
                other = std::max(other, preferred.width);
            }
        }
    } else {
        for (auto& preferred : preferred_sizes) {
            major.push_back(preferred.width);
            if (preferred.height >= Widget::DIM_GROW) {
                num_other_maxgrow_items += 1;
            } else {
                other = std::max(other, preferred.height);
            }
        }
    }

    if (other == 0 && num_other_maxgrow_items > 0) {
        other = Widget::DIM_GROW;
    }

    if (minor) {
        *minor = other;
    }
    return major;
}

std::vector<std::vector<std::shared_ptr<Widget>>> CalcColumns(
        int num_cols, const std::vector<std::shared_ptr<Widget>>& children) {
    std::vector<std::vector<std::shared_ptr<Widget>>> columns(num_cols);
    int col = 0;
    for (auto& child : children) {
        columns[col++].push_back(child);
        if (col >= num_cols) {
            col = 0;
        }
    }
    return columns;
}

std::vector<Size> CalcColumnSizes(
        const std::vector<std::vector<std::shared_ptr<Widget>>>& columns,
        const Theme& theme) {
    std::vector<Size> sizes;
    sizes.reserve(columns.size());

    for (auto& col : columns) {
        int w = 0, h = 0;
        for (auto& widget : col) {
            auto preferred = widget->CalcPreferredSize(theme);
            w = std::max(w, preferred.width);
            h += preferred.height;
        }
        sizes.push_back(Size(w, h));
    }

    return sizes;
}

}  // namespace

// ----------------------------------------------------------------------------
Margins::Margins() : left(0), top(0), right(0), bottom(0) {}
Margins::Margins(int px) : left(px), top(px), right(px), bottom(px) {}
Margins::Margins(int horiz_px, int vert_px)
    : left(horiz_px), top(vert_px), right(horiz_px), bottom(vert_px) {}
Margins::Margins(int left_px, int top_px, int right_px, int bottom_px)
    : left(left_px), top(top_px), right(right_px), bottom(bottom_px) {}

int Margins::GetHoriz() const { return this->left + this->right; }

int Margins::GetVert() const { return this->top + this->bottom; }

// ----------------------------------------------------------------------------
struct Layout1D::Impl {
    Layout1D::Dir dir_;
    int spacing_;
    Margins margins_;
};

void Layout1D::debug_PrintPreferredSizes(Layout1D* layout,
                                         const Theme& theme,
                                         int depth /*= 0*/) {
    static const char spaces[21] = "                    ";
    const char* indent = spaces + (20 - 3 * depth);
    auto pref_total = layout->CalcPreferredSize(theme);
    std::cout << "[debug] " << indent << "Layout1D ("
              << (layout->impl_->dir_ == Layout1D::VERT ? "VERT" : "HORIZ")
              << "): pref: (" << pref_total.width << ", " << pref_total.height
              << ")" << std::endl;
    std::cout << "[debug] " << indent << "spacing: " << layout->impl_->spacing_
              << ", margins: (l:" << layout->impl_->margins_.left
              << ", t:" << layout->impl_->margins_.top
              << ", r:" << layout->impl_->margins_.right
              << ", b:" << layout->impl_->margins_.bottom << ")" << std::endl;
    for (size_t i = 0; i < layout->GetChildren().size(); ++i) {
        auto child = layout->GetChildren()[i];
        auto pref = child->CalcPreferredSize(theme);
        std::cout << "[debug] " << indent << "i: " << i << " (" << pref.width
                  << ", " << pref.height << ")" << std::endl;
        Layout1D* child_layout = dynamic_cast<Layout1D*>(child.get());
        if (child_layout) {
            debug_PrintPreferredSizes(child_layout, theme, depth + 1);
        }
        VGrid* vgrid = dynamic_cast<VGrid*>(child.get());
        if (vgrid) {
            const char* grid_indent = spaces + (20 - 3 * (depth + 1));
            std::cout << "[debug] " << grid_indent
                      << "VGrid: spacing: " << vgrid->GetSpacing()
                      << ", margins: (l:" << vgrid->GetMargins().left
                      << ", t:" << vgrid->GetMargins().top
                      << ", r:" << vgrid->GetMargins().right
                      << ", b:" << vgrid->GetMargins().bottom << ")"
                      << std::endl;
            for (size_t i = 0; i < vgrid->GetChildren().size(); ++i) {
                auto e = vgrid->GetChildren()[i];
                auto pref = e->CalcPreferredSize(theme);
                std::cout << "[debug] " << grid_indent << "i: " << i << " ("
                          << pref.width << ", " << pref.height << ")"
                          << std::endl;
            }
        }
    }
}

Layout1D::Fixed::Fixed(int size, Dir dir) : size_(size), dir_(dir) {}

Size Layout1D::Fixed::CalcPreferredSize(const Theme& theme) const {
    if (dir_ == VERT) {
        return {0, size_};
    }

    return {size_, 0};
}

Size Layout1D::Stretch::CalcPreferredSize(const Theme& theme) const {
    return Size(0, 0);
}

Layout1D::Layout1D(Dir dir,
                   int spacing,
                   const Margins& margins,
                   const std::vector<std::shared_ptr<Widget>>& children)
    : Super(children), impl_(new Layout1D::Impl()) {
    impl_->dir_ = dir;
    impl_->spacing_ = spacing;
    impl_->margins_ = margins;
}

Layout1D::~Layout1D() {}

int Layout1D::GetSpacing() const { return impl_->spacing_; }
const Margins& Layout1D::GetMargins() const { return impl_->margins_; }
Margins& Layout1D::GetMutableMargins() { return impl_->margins_; }

void Layout1D::AddFixed(int size) {
    AddChild(std::make_shared<Fixed>(size, impl_->dir_));
}

void Layout1D::AddStretch() { AddChild(std::make_shared<Stretch>()); }

Size Layout1D::CalcPreferredSize(const Theme& theme) const {
    int minor;
    std::vector<int> major =
            CalcMajor(theme, impl_->dir_, GetChildren(), &minor);

    int total_spacing = impl_->spacing_ * (major.size() - 1);
    int major_size = 0;
    for (auto& size : major) {
        major_size += size;
    }

    if (impl_->dir_ == VERT) {
        return Size(minor + impl_->margins_.GetHoriz(),
                    major_size + impl_->margins_.GetVert() + total_spacing);
    } else {
        return Size(major_size + impl_->margins_.GetHoriz() + total_spacing,
                    minor + impl_->margins_.GetVert());
    }
}

void Layout1D::Layout(const Theme& theme) {
    auto frame = GetFrame();
    auto& children = GetChildren();
    std::vector<int> major = CalcMajor(theme, impl_->dir_, children, nullptr);
    int total = 0, num_stretch = 0, num_grow = 0;
    for (auto& mj : major) {
        total += mj;
        if (mj <= 0) {
            num_stretch += 1;
        }
        if (mj >= Widget::DIM_GROW) {
            num_grow += 1;
        }
    }
    int frame_size = (impl_->dir_ == VERT ? frame.height : frame.width);
    auto total_extra =
            frame_size - total - impl_->spacing_ * int(major.size() - 1);
    if (num_stretch > 0 && frame_size > total) {
        auto stretch = total_extra / num_stretch;
        auto leftover_stretch = total_extra - stretch * num_stretch;
        for (size_t i = 0; i < major.size(); ++i) {
            if (major[i] <= 0) {
                major[i] = stretch;
                if (leftover_stretch > 0) {
                    major[i] += 1;
                    leftover_stretch -= 1;
                }
            }
        }
    } else if (num_grow > 0 && frame_size < total) {
        auto total_excess = total - (frame_size - impl_->margins_.GetVert() -
                                     impl_->spacing_ * (major.size() - 1));
        auto excess = total_excess / num_grow;
        auto leftover = total_excess - excess * num_stretch;
        for (size_t i = 0; i < major.size(); ++i) {
            if (major[i] >= Widget::DIM_GROW) {
                major[i] -= excess;
                if (leftover > 0) {
                    major[i] -= 1;
                    leftover -= 1;
                }
            }
        }
    }

    int x = frame.GetLeft() + impl_->margins_.left;
    int y = frame.GetTop() + impl_->margins_.top;
    if (impl_->dir_ == VERT) {
        int minor = frame.width - impl_->margins_.GetHoriz();
        for (size_t i = 0; i < children.size(); ++i) {
            children[i]->SetFrame(Rect(x, y, minor, major[i]));
            y += major[i] + impl_->spacing_;
        }
    } else {
        int minor = frame.height - impl_->margins_.GetVert();
        for (size_t i = 0; i < children.size(); ++i) {
            children[i]->SetFrame(Rect(x, y, major[i], minor));
            x += major[i] + impl_->spacing_;
        }
    }

    Super::Layout(theme);
}

// ----------------------------------------------------------------------------
std::shared_ptr<Layout1D::Fixed> Vert::MakeFixed(int size) {
    return std::make_shared<Layout1D::Fixed>(size, VERT);
}

std::shared_ptr<Layout1D::Stretch> Vert::MakeStretch() {
    return std::make_shared<Layout1D::Stretch>();
}

Vert::Vert() : Layout1D(VERT, 0, Margins(), {}) {}

Vert::Vert(int spacing /*= 0*/, const Margins& margins /*= Margins()*/)
    : Layout1D(VERT, spacing, margins, {}) {}

Vert::Vert(int spacing,
           const Margins& margins,
           const std::vector<std::shared_ptr<Widget>>& children)
    : Layout1D(VERT, spacing, margins, children) {}

Vert::~Vert() {}

// ----------------------------------------------------------------------------
struct CollapsableVert::Impl {
    std::string id_;
    std::string text_;
    bool is_open_ = true;
};

CollapsableVert::CollapsableVert(const char* text)
    : CollapsableVert(text, 0, Margins()) {}

CollapsableVert::CollapsableVert(const char* text,
                                 int spacing,
                                 const Margins& margins /*= Margins()*/)
    : Vert(spacing, margins), impl_(new CollapsableVert::Impl()) {
    static int g_next_id = 1;

    impl_->text_ = text;

    std::stringstream s;
    s << text << "##collapsing" << g_next_id++;
    impl_->id_ = s.str();
}

CollapsableVert::~CollapsableVert() {}

void CollapsableVert::SetIsOpen(bool is_open) { impl_->is_open_ = is_open; }

Size CollapsableVert::CalcPreferredSize(const Theme& theme) const {
    auto* font = ImGui::GetFont();
    auto padding = ImGui::GetStyle().FramePadding;
    int text_height =
            std::ceil(ImGui::GetTextLineHeightWithSpacing() + 2 * padding.y);
    int text_width =
            std::ceil(font->CalcTextSizeA(theme.font_size, FLT_MAX, FLT_MAX,
                                          impl_->text_.c_str())
                              .x);

    auto pref = Super::CalcPreferredSize(theme);
    if (!impl_->is_open_) {
        pref.height = 0;
    }

    auto& margins = GetMargins();
    return Size(std::max(text_width, pref.width) + margins.GetHoriz(),
                text_height + pref.height + margins.GetVert());
}

void CollapsableVert::Layout(const Theme& theme) {
    auto padding = ImGui::GetStyle().FramePadding;
    int text_height =
            std::ceil(ImGui::GetTextLineHeightWithSpacing() + 2 * padding.y);

    auto& margins = GetMutableMargins();
    auto orig_top = margins.top;
    margins.top = orig_top + text_height;

    Super::Layout(theme);

    margins.top = orig_top;
}

Widget::DrawResult CollapsableVert::Draw(const DrawContext& context) {
    auto result = Widget::DrawResult::NONE;
    bool was_open = impl_->is_open_;

    auto& frame = GetFrame();
    ImGui::SetCursorPos(
            ImVec2(frame.x - context.uiOffsetX, frame.y - context.uiOffsetY));
    ImGui::PushItemWidth(frame.width);

    auto padding = ImGui::GetStyle().FramePadding;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, padding.y));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered,
                          colorToImgui(context.theme.button_hover_color));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive,
                          colorToImgui(context.theme.button_active_color));

    ImGui::SetNextTreeNodeOpen(impl_->is_open_);
    if (ImGui::TreeNode(impl_->id_.c_str())) {
        result = Super::Draw(context);
        ImGui::TreePop();
        impl_->is_open_ = true;
    } else {
        impl_->is_open_ = false;
    }

    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar();
    ImGui::PopItemWidth();

    if (impl_->is_open_ != was_open) {
        return DrawResult::RELAYOUT;
    }
    return result;
}
// ----------------------------------------------------------------------------
std::shared_ptr<Layout1D::Fixed> Horiz::MakeFixed(int size) {
    return std::make_shared<Layout1D::Fixed>(size, HORIZ);
}

std::shared_ptr<Layout1D::Stretch> Horiz::MakeStretch() {
    return std::make_shared<Layout1D::Stretch>();
}

std::shared_ptr<Horiz> Horiz::MakeCentered(std::shared_ptr<Widget> w) {
    return std::make_shared<Horiz>(
            0, Margins(),
            std::vector<std::shared_ptr<Widget>>(
                    {Horiz::MakeStretch(), w, Horiz::MakeStretch()}));
}

Horiz::Horiz() : Layout1D(HORIZ, 0, Margins(), {}) {}

Horiz::Horiz(int spacing /*= 0*/, const Margins& margins /*= Margins()*/)
    : Layout1D(HORIZ, spacing, margins, {}) {}

Horiz::Horiz(int spacing,
             const Margins& margins,
             const std::vector<std::shared_ptr<Widget>>& children)
    : Layout1D(HORIZ, spacing, margins, children) {}

Horiz::~Horiz() {}

// ----------------------------------------------------------------------------
struct VGrid::Impl {
    int num_cols_;
    int spacing_;
    Margins margins_;
};

VGrid::VGrid(int num_cols,
             int spacing /*= 0*/,
             const Margins& margins /*= Margins()*/)
    : impl_(new VGrid::Impl()) {
    impl_->num_cols_ = num_cols;
    impl_->spacing_ = spacing;
    impl_->margins_ = margins;
}

VGrid::~VGrid() {}

int VGrid::GetSpacing() const { return impl_->spacing_; }
const Margins& VGrid::GetMargins() const { return impl_->margins_; }

Size VGrid::CalcPreferredSize(const Theme& theme) const {
    auto columns = CalcColumns(impl_->num_cols_, GetChildren());
    auto column_sizes = CalcColumnSizes(columns, theme);

    int width = 0, height = 0;
    for (size_t i = 0; i < column_sizes.size(); ++i) {
        auto& sz = column_sizes[i];
        width += sz.width;
        auto v_spacing = (columns[i].size() - 1) * impl_->spacing_;
        height = std::max(height, sz.height) + v_spacing;
    }
    width += (column_sizes.size() - 1) * impl_->spacing_;
    width = std::max(width, 0);  // in case width or height has no items
    height = std::max(height, 0);

    return Size(width + impl_->margins_.left + impl_->margins_.right,
                height + impl_->margins_.top + impl_->margins_.bottom);
}

void VGrid::Layout(const Theme& theme) {
    auto columns = CalcColumns(impl_->num_cols_, GetChildren());
    auto column_sizes = CalcColumnSizes(columns, theme);

    // Shrink columns that are too big.
    // TODO: right now this only handles DIM_GROW columns; extend to
    //       proportionally shrink columns that together add up to too much.
    //       Probably should figure out how to reuse for other layouts.
    auto& frame = GetFrame();
    const int layout_width =
            frame.width - impl_->margins_.left - impl_->margins_.right;
    int wanted_width = 0;
    int total_not_growing_width = 0;
    int num_growing = 0;
    for (auto& sz : column_sizes) {
        wanted_width += sz.width;
        if (sz.width < DIM_GROW) {
            total_not_growing_width += sz.width;
        } else {
            num_growing += 1;
        }
    }
    if (wanted_width > layout_width && num_growing > 0) {
        int growing_size =
                (layout_width - total_not_growing_width) / num_growing;
        if (growing_size < 0) {
            growing_size = layout_width / num_growing;
        }
        for (auto& sz : column_sizes) {
            if (sz.width >= DIM_GROW) {
                sz.width = growing_size;
            }
        }
    }

    // Adjust the columns for spacing. The code above adjusted width
    // without taking intra-element spacing, so do that here.
    int leftHalf = int(std::floor(float(impl_->spacing_) / 2.0));
    int rightHalf = int(std::ceil(float(impl_->spacing_) / 2.0));
    for (size_t i = 0; i < column_sizes.size() - 1; ++i) {
        column_sizes[i].width -= leftHalf;
        column_sizes[i + 1].width -= rightHalf;
    }

    // Do the layout
    int x = frame.GetLeft() + impl_->margins_.left;
    for (size_t i = 0; i < columns.size(); ++i) {
        int y = frame.GetTop() + impl_->margins_.top;
        for (auto& w : columns[i]) {
            auto preferred = w->CalcPreferredSize(theme);
            w->SetFrame(Rect(x, y, column_sizes[i].width, preferred.height));
            y += preferred.height + impl_->spacing_;
        }
        x += column_sizes[i].width + impl_->spacing_;
    }

    Super::Layout(theme);
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
