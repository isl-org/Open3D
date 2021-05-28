// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2021 www.open3d.org
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

#include "open3d/visualization/gui/Application.h"
#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

namespace {

std::vector<int> CalcMajor(const LayoutContext& context,
                           const Widget::Constraints& constraints,
                           Layout1D::Dir dir,
                           const std::vector<std::shared_ptr<Widget>>& children,
                           int* minor = nullptr) {
    std::vector<Size> preferred_sizes;
    preferred_sizes.reserve(children.size());
    for (auto& child : children) {
        preferred_sizes.push_back(
                child->CalcPreferredSize(context, constraints));
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
        const LayoutContext& context,
        const Widget::Constraints& constraints) {
    std::vector<Size> sizes;
    sizes.reserve(columns.size());

    for (auto& col : columns) {
        int w = 0, h = 0;
        for (auto& widget : col) {
            auto preferred = widget->CalcPreferredSize(context, constraints);
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
    int minor_axis_size_ = Widget::DIM_GROW;
};

void Layout1D::debug_PrintPreferredSizes(Layout1D* layout,
                                         const LayoutContext& context,
                                         const Constraints& constraints,
                                         int depth /*= 0*/) {
    static const char spaces[21] = "                    ";
    const char* indent = spaces + (20 - 3 * depth);
    auto pref_total = layout->CalcPreferredSize(context, constraints);
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
        auto pref = child->CalcPreferredSize(context, constraints);
        std::cout << "[debug] " << indent << "i: " << i << " (" << pref.width
                  << ", " << pref.height << ")" << std::endl;
        Layout1D* child_layout = dynamic_cast<Layout1D*>(child.get());
        if (child_layout) {
            debug_PrintPreferredSizes(child_layout, context, constraints,
                                      depth + 1);
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
                auto pref = e->CalcPreferredSize(context, constraints);
                std::cout << "[debug] " << grid_indent << "i: " << i << " ("
                          << pref.width << ", " << pref.height << ")"
                          << std::endl;
            }
        }
    }
}

Layout1D::Fixed::Fixed(int size, Dir dir) : size_(size), dir_(dir) {}

Size Layout1D::Fixed::CalcPreferredSize(const LayoutContext& context,
                                        const Constraints& constraints) const {
    if (dir_ == VERT) {
        return {0, size_};
    }

    return {size_, 0};
}

Size Layout1D::Stretch::CalcPreferredSize(
        const LayoutContext& context, const Constraints& constraints) const {
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

void Layout1D::SetSpacing(int spacing) { impl_->spacing_ = spacing; }
void Layout1D::SetMargins(const Margins& margins) { impl_->margins_ = margins; }

void Layout1D::AddFixed(int size) {
    AddChild(std::make_shared<Fixed>(size, impl_->dir_));
}

int Layout1D::GetMinorAxisPreferredSize() const {
    return impl_->minor_axis_size_;
}

void Layout1D::SetMinorAxisPreferredSize(int size) {
    impl_->minor_axis_size_ = size;
}

void Layout1D::AddStretch() { AddChild(std::make_shared<Stretch>()); }

Size Layout1D::CalcPreferredSize(const LayoutContext& context,
                                 const Constraints& constraints) const {
    int minor;
    std::vector<int> major =
            CalcMajor(context, constraints, impl_->dir_, GetChildren(), &minor);
    if (impl_->minor_axis_size_ < Widget::DIM_GROW) {
        minor = impl_->minor_axis_size_;
    }

    int total_spacing = impl_->spacing_ * std::max(0, int(major.size()) - 1);
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

void Layout1D::Layout(const LayoutContext& context) {
    auto frame = GetFrame();
    Constraints constraints;
    if (impl_->dir_ == VERT) {
        constraints.width =
                frame.width - impl_->margins_.left - impl_->margins_.right;
    } else {
        constraints.height =
                frame.height - impl_->margins_.top - impl_->margins_.bottom;
    }
    auto& children = GetChildren();
    std::vector<int> major =
            CalcMajor(context, constraints, impl_->dir_, children, nullptr);
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
    int frame_size;
    if (impl_->dir_ == VERT) {
        frame_size = frame.height - impl_->margins_.GetVert();
    } else {
        frame_size = frame.width - impl_->margins_.GetHoriz();
    }
    int total_spacing = impl_->spacing_ * std::max(0, int(major.size()) - 1);
    auto total_extra = frame_size - total - total_spacing;
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
    } else if (frame_size < total) {
        int n_shrinkable = num_grow;
        if (impl_->dir_ == VERT) {
            for (auto child : GetChildren()) {
                if (std::dynamic_pointer_cast<ScrollableVert>(child)) {
                    n_shrinkable++;
                }
            }
        }
        if (n_shrinkable > 0) {
            auto total_excess = total - (frame_size - total_spacing);
            auto excess = total_excess / n_shrinkable;
            auto leftover = total_excess - excess * num_grow;
            for (size_t i = 0; i < major.size(); ++i) {
                if (major[i] >= Widget::DIM_GROW ||
                    (impl_->dir_ == VERT &&
                     std::dynamic_pointer_cast<ScrollableVert>(
                             GetChildren()[i]) != nullptr)) {
                    major[i] -= excess;
                    if (leftover > 0) {
                        major[i] -= 1;
                        leftover -= 1;
                    }
                }
            }
        }
    }

    int x = frame.GetLeft() + impl_->margins_.left;
    int y = frame.GetTop() + impl_->margins_.top;
    if (impl_->dir_ == VERT) {
        int minor = frame.width - impl_->margins_.GetHoriz();
        for (size_t i = 0; i < children.size(); ++i) {
            int h = std::max(children[i]->CalcMinimumSize(context).height,
                             major[i]);
            children[i]->SetFrame(Rect(x, y, minor, h));
            y += major[i] + impl_->spacing_;
        }
    } else {
        int minor = frame.height - impl_->margins_.GetVert();
        for (size_t i = 0; i < children.size(); ++i) {
            children[i]->SetFrame(Rect(x, y, major[i], minor));
            x += major[i] + impl_->spacing_;
        }
    }

    Super::Layout(context);
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

int Vert::GetPreferredWidth() const { return GetMinorAxisPreferredSize(); }
void Vert::SetPreferredWidth(int w) { SetMinorAxisPreferredSize(w); }

// ----------------------------------------------------------------------------
struct CollapsableVert::Impl {
    std::string id_;
    std::string text_;
    FontId font_id_ = Application::DEFAULT_FONT_ID;
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
    impl_->id_ = impl_->text_ + "##collapsing_" + std::to_string(g_next_id++);
}

CollapsableVert::~CollapsableVert() {}

void CollapsableVert::SetIsOpen(bool is_open) { impl_->is_open_ = is_open; }

bool CollapsableVert::GetIsOpen() { return impl_->is_open_; }

FontId CollapsableVert::GetFontId() const { return impl_->font_id_; }

void CollapsableVert::SetFontId(FontId font_id) { impl_->font_id_ = font_id; }

Size CollapsableVert::CalcPreferredSize(const LayoutContext& context,
                                        const Constraints& constraints) const {
    // Only push the font for the label
    ImGui::PushFont((ImFont*)context.fonts.GetFont(impl_->font_id_));
    auto* font = ImGui::GetFont();
    auto padding = ImGui::GetStyle().FramePadding;
    int text_height = int(
            std::ceil(ImGui::GetTextLineHeightWithSpacing() + 2 * padding.y));
    int text_width =
            int(std::ceil(font->CalcTextSizeA(font->FontSize, FLT_MAX, FLT_MAX,
                                              impl_->text_.c_str())
                                  .x));
    ImGui::PopFont();  // back to default font for layout sizing

    auto pref = Super::CalcPreferredSize(context, constraints);
    if (!impl_->is_open_) {
        pref.height = 0;
    }

    auto& margins = GetMargins();
    return Size(std::max(text_width, pref.width) + margins.GetHoriz(),
                text_height + pref.height + margins.GetVert());
}

void CollapsableVert::Layout(const LayoutContext& context) {
    ImGui::PushFont((ImFont*)context.fonts.GetFont(impl_->font_id_));
    auto padding = ImGui::GetStyle().FramePadding;
    int text_height = int(
            std::ceil(ImGui::GetTextLineHeightWithSpacing() + 2 * padding.y));
    auto& margins = GetMutableMargins();
    auto orig_top = margins.top;
    margins.top = orig_top + text_height;
    ImGui::PopFont();

    Super::Layout(context);

    margins.top = orig_top;
}

Widget::DrawResult CollapsableVert::Draw(const DrawContext& context) {
    auto result = Widget::DrawResult::NONE;
    bool was_open = impl_->is_open_;

    auto& frame = GetFrame();
    ImGui::SetCursorScreenPos(
            ImVec2(float(frame.x), float(frame.y) - ImGui::GetScrollY()));
    ImGui::PushItemWidth(float(frame.width));

    auto padding = ImGui::GetStyle().FramePadding;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, padding.y));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered,
                          colorToImgui(context.theme.button_hover_color));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive,
                          colorToImgui(context.theme.button_active_color));

    ImGui::SetNextTreeNodeOpen(impl_->is_open_);
    ImGui::PushFont((ImFont*)context.fonts.GetFont(impl_->font_id_));
    bool node_clicked = ImGui::TreeNode(impl_->id_.c_str());
    ImGui::PopFont();
    if (node_clicked) {
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
struct ScrollableVert::Impl {
    ImGuiID id_;
};

ScrollableVert::ScrollableVert() : ScrollableVert(0, Margins(), {}) {}

ScrollableVert::ScrollableVert(int spacing /*= 0*/,
                               const Margins& margins /*= Margins()*/)
    : ScrollableVert(spacing, margins, {}) {}

ScrollableVert::ScrollableVert(
        int spacing,
        const Margins& margins,
        const std::vector<std::shared_ptr<Widget>>& children)
    : Vert(spacing, margins, children), impl_(new ScrollableVert::Impl) {
    static int g_next_id = 1;
    impl_->id_ = g_next_id++;
}

ScrollableVert::~ScrollableVert() {}

Widget::DrawResult ScrollableVert::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    ImGui::SetCursorScreenPos(
            ImVec2(float(frame.x), float(frame.y) - ImGui::GetScrollY()));
    ImGui::PushStyleColor(ImGuiCol_FrameBg,
                          ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
    ImGui::PushStyleColor(ImGuiCol_Border,
                          colorToImgui(Color(0.0f, 0.0f, 0.0f, 0.0f)));
    ImGui::PushStyleColor(ImGuiCol_BorderShadow,
                          colorToImgui(Color(0.0f, 0.0f, 0.0f, 0.0f)));

    ImGui::BeginChildFrame(impl_->id_, ImVec2(frame.width, frame.height));
    auto result = Super::Draw(context);
    ImGui::EndChildFrame();

    ImGui::PopStyleColor(3);

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

int Horiz::GetPreferredHeight() const { return GetMinorAxisPreferredSize(); }
void Horiz::SetPreferredHeight(int h) { SetMinorAxisPreferredSize(h); }

// ----------------------------------------------------------------------------
struct VGrid::Impl {
    int num_cols_;
    int spacing_;
    Margins margins_;
    int preferred_width_ = Widget::DIM_GROW;
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

int VGrid::GetPreferredWidth() const { return impl_->preferred_width_; }
void VGrid::SetPreferredWidth(int w) { impl_->preferred_width_ = w; }

Size VGrid::CalcPreferredSize(const LayoutContext& context,
                              const Constraints& constraints) const {
    auto columns = CalcColumns(impl_->num_cols_, GetChildren());
    auto column_sizes = CalcColumnSizes(columns, context, constraints);

    int width = 0, height = 0;
    for (size_t i = 0; i < column_sizes.size(); ++i) {
        auto& sz = column_sizes[i];
        width += sz.width;
        height = std::max(height, sz.height);
        if (i < column_sizes.size() - 1) {
            auto v_spacing = (int(columns[i].size()) - 1) * impl_->spacing_;
            height += v_spacing;
        }
    }
    width += (int(column_sizes.size()) - 1) * impl_->spacing_;
    width = std::max(width, 0);  // in case width or height has no items
    height = std::max(height, 0);

    if (impl_->preferred_width_ < Widget::DIM_GROW) {
        width = impl_->preferred_width_;
    } else {
        width = width + impl_->margins_.left + impl_->margins_.right;
    }

    return Size(width, height + impl_->margins_.top + impl_->margins_.bottom);
}

void VGrid::Layout(const LayoutContext& context) {
    auto& frame = GetFrame();
    const int layout_width =
            frame.width - impl_->margins_.left - impl_->margins_.right;
    Constraints constraints;
    constraints.width = layout_width;

    auto columns = CalcColumns(impl_->num_cols_, GetChildren());
    auto column_sizes = CalcColumnSizes(columns, context, constraints);

    // Shrink columns that are too big.
    // TODO: right now this only handles DIM_GROW columns; extend to
    //       proportionally shrink columns that together add up to too much.
    //       Probably should figure out how to reuse for other layouts.
    int grow_size = constraints.width;
    int wanted_width = 0;
    int total_not_growing_width = 0;
    int num_growing = 0;
    for (auto& sz : column_sizes) {
        wanted_width += sz.width;
        if (sz.width < grow_size) {
            total_not_growing_width += sz.width;
        } else {
            num_growing += 1;
        }
    }
    if (wanted_width > layout_width && num_growing > 0) {
        int total_spacing = (int(columns.size()) - 1) * impl_->spacing_;
        int growing_size =
                (layout_width - total_spacing - total_not_growing_width) /
                num_growing;
        if (growing_size < 0) {
            growing_size = layout_width / num_growing;
        }
        for (auto& sz : column_sizes) {
            if (sz.width >= grow_size) {
                sz.width = growing_size;
            }
        }
    } else {
        // Just adjust the columns for spacing.
        int leftHalf = int(std::floor(float(impl_->spacing_) / 2.0));
        int rightHalf = int(std::ceil(float(impl_->spacing_) / 2.0));
        for (size_t i = 0; i < column_sizes.size() - 1; ++i) {
            column_sizes[i].width -= leftHalf;
            column_sizes[i + 1].width -= rightHalf;
        }
    }

    // Do the layout
    int x = frame.GetLeft() + impl_->margins_.left;
    for (size_t i = 0; i < columns.size(); ++i) {
        Constraints constraints;
        constraints.width = column_sizes[i].width;
        int y = frame.GetTop() + impl_->margins_.top;
        for (auto& w : columns[i]) {
            auto preferred = w->CalcPreferredSize(context, constraints);
            w->SetFrame(Rect(x, y, column_sizes[i].width, preferred.height));
            y += preferred.height + impl_->spacing_;
        }
        x += column_sizes[i].width + impl_->spacing_;
    }

    Super::Layout(context);
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
