// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

#include "open3d/visualization/gui/TreeView.h"

#include <imgui.h>

#include <cmath>
#include <list>
#include <sstream>
#include <unordered_map>

#include "open3d/visualization/gui/Checkbox.h"
#include "open3d/visualization/gui/ColorEdit.h"
#include "open3d/visualization/gui/Label.h"
#include "open3d/visualization/gui/NumberEdit.h"
#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

struct CheckableTextTreeCell::Impl {
    std::shared_ptr<Checkbox> checkbox_;
    std::shared_ptr<Label> label_;
};

CheckableTextTreeCell::CheckableTextTreeCell(
        const char *text, bool is_checked, std::function<void(bool)> on_toggled)
    : impl_(new CheckableTextTreeCell::Impl()) {
    // We don't want any text in the checkbox, but passing "" seems to make it
    // not toggle, so we need to pass in something. This way it will just be
    // extra spacing.
    impl_->checkbox_ = std::make_shared<Checkbox>(" ");
    impl_->checkbox_->SetChecked(is_checked);
    impl_->checkbox_->SetOnChecked(on_toggled);
    impl_->label_ = std::make_shared<Label>(text);
    AddChild(impl_->checkbox_);
    AddChild(impl_->label_);
}

CheckableTextTreeCell::~CheckableTextTreeCell() {}

std::shared_ptr<Checkbox> CheckableTextTreeCell::GetCheckbox() {
    return impl_->checkbox_;
}

std::shared_ptr<Label> CheckableTextTreeCell::GetLabel() {
    return impl_->label_;
}

Size CheckableTextTreeCell::CalcPreferredSize(const Theme &theme) const {
    auto check_pref = impl_->checkbox_->CalcPreferredSize(theme);
    auto label_pref = impl_->label_->CalcPreferredSize(theme);
    return Size(check_pref.width + label_pref.width,
                std::max(check_pref.height, label_pref.height));
}

void CheckableTextTreeCell::Layout(const Theme &theme) {
    auto &frame = GetFrame();
    auto check_width = impl_->checkbox_->CalcPreferredSize(theme).width;
    impl_->checkbox_->SetFrame(
            Rect(frame.x, frame.y, check_width, frame.height));
    auto x = impl_->checkbox_->GetFrame().GetRight();
    impl_->label_->SetFrame(
            Rect(x, frame.y, frame.GetRight() - x, frame.height));
}

// ----------------------------------------------------------------------------
struct LUTTreeCell::Impl {
    std::shared_ptr<Checkbox> checkbox_;
    std::shared_ptr<Label> label_;
    std::shared_ptr<ColorEdit> color_;
    float color_width_percent = 0.2f;
};

LUTTreeCell::LUTTreeCell(const char *text,
                         bool is_checked,
                         const Color &color,
                         std::function<void(bool)> on_enabled,
                         std::function<void(const Color &)> on_color_changed)
    : impl_(new LUTTreeCell::Impl()) {
    // We don't want any text in the checkbox, but passing "" seems to make it
    // not toggle, so we need to pass in something. This way it will just be
    // extra spacing.
    impl_->checkbox_ = std::make_shared<Checkbox>(" ");
    impl_->checkbox_->SetChecked(is_checked);
    impl_->checkbox_->SetOnChecked(on_enabled);
    impl_->label_ = std::make_shared<Label>(text);
    impl_->color_ = std::make_shared<ColorEdit>();
    impl_->color_->SetValue(color);
    impl_->color_->SetOnValueChanged(on_color_changed);
    AddChild(impl_->checkbox_);
    AddChild(impl_->label_);
    AddChild(impl_->color_);
}

LUTTreeCell::~LUTTreeCell() {}

std::shared_ptr<Checkbox> LUTTreeCell::GetCheckbox() {
    return impl_->checkbox_;
}

std::shared_ptr<Label> LUTTreeCell::GetLabel() { return impl_->label_; }

std::shared_ptr<ColorEdit> LUTTreeCell::GetColorEdit() { return impl_->color_; }

Size LUTTreeCell::CalcPreferredSize(const Theme &theme) const {
    auto check_pref = impl_->checkbox_->CalcPreferredSize(theme);
    auto label_pref = impl_->label_->CalcPreferredSize(theme);
    auto color_pref = impl_->color_->CalcPreferredSize(theme);
    return Size(check_pref.width + label_pref.width + color_pref.width,
                std::max(check_pref.height,
                         std::max(label_pref.height, color_pref.height)));
}

void LUTTreeCell::Layout(const Theme &theme) {
    auto em = theme.font_size;
    auto &frame = GetFrame();
    auto check_width = impl_->checkbox_->CalcPreferredSize(theme).width;
    auto color_width =
            int(std::ceil(impl_->color_width_percent * float(frame.width)));
    auto min_color_width = 8 * theme.font_size;
    color_width = std::max(min_color_width, color_width);
    if (frame.width - (color_width + check_width) < 8 * em) {
        color_width = frame.width - check_width - 8 * em;
    }
    impl_->checkbox_->SetFrame(
            Rect(frame.x, frame.y, check_width, frame.height));
    impl_->color_->SetFrame(Rect(frame.GetRight() - color_width, frame.y,
                                 color_width, frame.height));
    auto x = impl_->checkbox_->GetFrame().GetRight();
    impl_->label_->SetFrame(
            Rect(x, frame.y, impl_->color_->GetFrame().x - x, frame.height));
}

// ----------------------------------------------------------------------------
struct ColormapTreeCell::Impl {
    std::shared_ptr<NumberEdit> value_;
    std::shared_ptr<ColorEdit> color_;
};

ColormapTreeCell::ColormapTreeCell(
        double value,
        const Color &color,
        std::function<void(double)> on_value_changed,
        std::function<void(const Color &)> on_color_changed)
    : impl_(new ColormapTreeCell::Impl()) {
    impl_->value_ = std::make_shared<NumberEdit>(NumberEdit::DOUBLE);
    impl_->value_->SetDecimalPrecision(3);
    impl_->value_->SetLimits(0.0, 1.0);
    impl_->value_->SetValue(value);
    impl_->value_->SetOnValueChanged(on_value_changed);
    impl_->color_ = std::make_shared<ColorEdit>();
    impl_->color_->SetValue(color);
    impl_->color_->SetOnValueChanged(on_color_changed);
    AddChild(impl_->value_);
    AddChild(impl_->color_);
}

ColormapTreeCell::~ColormapTreeCell() {}

std::shared_ptr<NumberEdit> ColormapTreeCell::GetNumberEdit() {
    return impl_->value_;
}

std::shared_ptr<ColorEdit> ColormapTreeCell::GetColorEdit() {
    return impl_->color_;
}

Size ColormapTreeCell::CalcPreferredSize(const Theme &theme) const {
    auto number_pref = impl_->value_->CalcPreferredSize(theme);
    auto color_pref = impl_->color_->CalcPreferredSize(theme);
    return Size(number_pref.width + color_pref.width,
                std::max(number_pref.height, color_pref.height));
}

void ColormapTreeCell::Layout(const Theme &theme) {
    auto &frame = GetFrame();
    auto number_pref = impl_->value_->CalcPreferredSize(theme);
    impl_->value_->SetFrame(
            Rect(frame.x, frame.y, number_pref.width, frame.height));
    auto x = impl_->value_->GetFrame().GetRight();
    impl_->color_->SetFrame(
            Rect(x, frame.y, frame.GetRight() - x, frame.height));
}

// ----------------------------------------------------------------------------
namespace {
static int g_treeview_id = 1;
}

struct TreeView::Impl {
    static TreeView::ItemId g_next_id;

    // Note: use std::list because pointers remain valid, unlike std::vector
    //       which will invalidate pointers when it resizes the underlying
    //       array
    struct Item {
        TreeView::ItemId id = -1;
        std::string id_string;
        std::shared_ptr<Widget> cell;
        Item *parent = nullptr;
        std::list<Item> children;
    };
    int id_;
    Item root_;
    std::unordered_map<TreeView::ItemId, Item *> id2item_;
    TreeView::ItemId selected_id_ = -1;
    bool can_select_parents_ = false;
    std::function<void(TreeView::ItemId)> on_selection_changed_;
};

TreeView::ItemId TreeView::Impl::g_next_id = 0;

TreeView::TreeView() : impl_(new TreeView::Impl()) {
    impl_->id_ = g_treeview_id++;
    impl_->root_.id = Impl::g_next_id++;
    impl_->id2item_[impl_->root_.id] = &impl_->root_;
}

TreeView::~TreeView() {}

TreeView::ItemId TreeView::GetRootItem() const { return impl_->root_.id; }

TreeView::ItemId TreeView::AddItem(ItemId parent_id,
                                   std::shared_ptr<Widget> w) {
    Impl::Item item;
    item.id = Impl::g_next_id++;
    // ImGUI uses the text to identify the item, create a ID string
    std::stringstream s;
    s << "treeview" << impl_->id_ << "item" << item.id;
    item.id_string = s.str();
    item.cell = w;

    Impl::Item *parent = &impl_->root_;
    auto parent_it = impl_->id2item_.find(parent_id);
    if (parent_it != impl_->id2item_.end()) {
        parent = parent_it->second;
    }
    item.parent = parent;
    parent->children.push_back(item);
    impl_->id2item_[item.id] = &parent->children.back();

    return item.id;
}

TreeView::ItemId TreeView::AddTextItem(ItemId parent_id, const char *text) {
    std::shared_ptr<Widget> w = std::make_shared<Label>(text);
    return AddItem(parent_id, w);
}

void TreeView::RemoveItem(ItemId item_id) {
    auto item_it = impl_->id2item_.find(item_id);
    if (item_it != impl_->id2item_.end()) {
        auto item = item_it->second;
        // Erase the item here, because RemoveItem(child) will also erase,
        // which will invalidate our iterator.
        impl_->id2item_.erase(item_it);

        // Remove children. Note that we can't use a foreach loop here,
        // because when we remove the item from its parent it will
        // invalidate the iterator to the current item that exists under
        // the hood, making `it++` not workable. So we use a while loop
        // instead. Because this is a list, we can erase from the front
        // in O(1).
        while (!item->children.empty()) {
            RemoveItem(item->children.front().id);
        }

        // Remove ourself from our parent's list of children
        if (item->parent) {
            for (auto sibling = item->parent->children.begin();
                 sibling != item->parent->children.end(); ++sibling) {
                if (sibling->id == item_id) {
                    item->parent->children.erase(sibling);
                    break;
                }
            }
        }
    }
}

void TreeView::Clear() {
    impl_->selected_id_ = -1;
    impl_->id2item_.clear();
    impl_->root_.children.clear();
}

std::shared_ptr<Widget> TreeView::GetItem(ItemId item_id) const {
    auto item_it = impl_->id2item_.find(item_id);
    if (item_it != impl_->id2item_.end()) {
        return item_it->second->cell;
    }
    return nullptr;
}

std::vector<TreeView::ItemId> TreeView::GetItemChildren(
        ItemId parent_id) const {
    std::vector<TreeView::ItemId> children;
    auto item_it = impl_->id2item_.find(parent_id);
    if (item_it != impl_->id2item_.end()) {
        auto *parent = item_it->second->parent;
        if (parent) {
            children.reserve(parent->children.size());
            for (auto &child : parent->children) {
                children.push_back(child.id);
            }
        }
    }
    return children;
}

bool TreeView::GetCanSelectItemsWithChildren() const {
    return impl_->can_select_parents_;
}

void TreeView::SetCanSelectItemsWithChildren(bool can_select) {
    impl_->can_select_parents_ = can_select;
}

TreeView::ItemId TreeView::GetSelectedItemId() const {
    if (impl_->selected_id_ < 0) {
        return impl_->root_.id;
    } else {
        return impl_->selected_id_;
    }
}

void TreeView::SetSelectedItemId(ItemId item_id) {
    impl_->selected_id_ = item_id;
}

void TreeView::SetOnSelectionChanged(
        std::function<void(ItemId)> on_selection_changed) {
    impl_->on_selection_changed_ = on_selection_changed;
}

Size TreeView::CalcPreferredSize(const Theme &theme) const {
    return Size(Widget::DIM_GROW, Widget::DIM_GROW);
}

void TreeView::Layout(const Theme &theme) {
    // Nothing to do here. We don't know the x position because of the
    // indentations, which also means we don't know the size. So we need
    // to defer layout to Draw().
}

Widget::DrawResult TreeView::Draw(const DrawContext &context) {
    auto result = Widget::DrawResult::NONE;
    auto &frame = GetFrame();

    DrawImGuiPushEnabledState();
    ImGui::SetCursorScreenPos(ImVec2(float(frame.x), float(frame.y)));

    // ImGUI's tree wants to highlight the row as the user moves over it.
    // There are several problems here. First, there seems to be a bug in
    // ImGUI where the highlight ignores the pushed item width and extends
    // to the end of the ImGUI-window (i.e. the topmost parent Widget). This
    // means the highlight extends into any margins we have. Not good. Second,
    // the highlight extends past the clickable area, which is misleading.
    // Third, no operating system has hover highlights like this, and it looks
    // really strange. I mean, you can see the cursor right over your text,
    // what do you need a highligh for? So make this highlight transparent.
    ImGui::PushStyleColor(ImGuiCol_HeaderActive,  // click-hold on item
                          colorToImgui(Color(0, 0, 0, 0)));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered,
                          colorToImgui(Color(0, 0, 0, 0)));

    ImGui::PushStyleColor(ImGuiCol_ChildBg,
                          colorToImgui(context.theme.tree_background_color));

    // ImGUI's tree is basically a layout in the parent ImGUI window.
    // Make this a child so it's all in a nice frame.
    ImGui::BeginChild(impl_->id_,
                      ImVec2(float(frame.width), float(frame.height)), true);

    Impl::Item *new_selection = nullptr;

    std::function<void(Impl::Item &)> DrawItem;
    DrawItem = [&DrawItem, this, &frame, &context, &new_selection,
                &result](Impl::Item &item) {
        int height = item.cell->CalcPreferredSize(context.theme).height;

        // ImGUI's tree doesn't seem to support selected items,
        // so we have to draw our own selection.
        if (item.id == impl_->selected_id_) {
            // Since we are in a child, the cursor is relative to the upper left
            // of the tree's frame. To draw directly to the window list we
            // need to the absolute coordinates (relative the OS window's
            // upper left)
            auto y = frame.y + ImGui::GetCursorPosY() - ImGui::GetScrollY();
            ImGui::GetWindowDrawList()->AddRectFilled(
                    ImVec2(float(frame.x), y),
                    ImVec2(float(frame.GetRight()), y + height),
                    colorToImguiRGBA(context.theme.tree_selected_color));
        }

        int flags = ImGuiTreeNodeFlags_DefaultOpen |
                    ImGuiTreeNodeFlags_AllowItemOverlap;
        if (impl_->can_select_parents_) {
            flags |= ImGuiTreeNodeFlags_OpenOnDoubleClick;
            flags |= ImGuiTreeNodeFlags_OpenOnArrow;
        }
        if (item.children.empty()) {
            flags |= ImGuiTreeNodeFlags_Leaf;
        }
        bool is_selectable =
                (item.children.empty() || impl_->can_select_parents_);
        auto DrawThis = [this, &tree_frame = frame, &context, &new_selection,
                         &result](TreeView::Impl::Item &item, int height,
                                  bool is_selectable) {
            ImGui::SameLine(0, 0);
            auto x = int(std::round(ImGui::GetCursorScreenPos().x));
            auto y = int(std::round(ImGui::GetCursorScreenPos().y));
            auto scroll_width = int(ImGui::GetStyle().ScrollbarSize);
            auto indent = x - tree_frame.x;
            item.cell->SetFrame(Rect(
                    x, y, tree_frame.width - indent - scroll_width, height));
            // Now that we know the frame we can finally layout. It would be
            // nice to not relayout until something changed, which would
            // usually work, unless the cell changes shape in response to
            // something, which would be a problem. So do it every time.
            item.cell->Layout(context.theme);

            ImGui::BeginGroup();
            auto this_result = item.cell->Draw(context);
            if (this_result == Widget::DrawResult::REDRAW) {
                result = Widget::DrawResult::REDRAW;
            }
            ImGui::EndGroup();

            if (ImGui::IsItemClicked() && is_selectable) {
                impl_->selected_id_ = item.id;
                new_selection = &item;
            }
        };

        if (ImGui::TreeNodeEx(item.id_string.c_str(), flags, "%s", "")) {
            DrawThis(item, height, is_selectable);

            for (auto &child : item.children) {
                DrawItem(child);
            }
            ImGui::TreePop();
        } else {
            DrawThis(item, height, is_selectable);
        }
    };
    for (auto &top : impl_->root_.children) {
        DrawItem(top);
    }

    ImGui::EndChild();

    ImGui::PopStyleColor(3);
    DrawImGuiPopEnabledState();

    // If the selection changed, handle the callback here, after we have
    // finished drawing, so that the callback is able to change the contents
    // of the tree if it wishes (which could cause a crash if done while
    // drawing, e.g. deleting the current item).
    if (new_selection) {
        if (impl_->on_selection_changed_) {
            impl_->on_selection_changed_(new_selection->id);
        }
        result = Widget::DrawResult::REDRAW;
    }

    return result;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
