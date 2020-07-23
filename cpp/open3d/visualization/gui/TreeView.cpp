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
#include <list>
#include <sstream>
#include <unordered_map>

#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

namespace open3d {
namespace visualization {
namespace gui {

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
        std::string text;
        Item *parent = nullptr;
        std::list<Item> children;
    };
    int id_;
    Item root_;
    std::unordered_map<TreeView::ItemId, Item *> id2item_;
    TreeView::ItemId selected_id_ = -1;
    bool can_select_parents_ = false;
    std::function<void(const char *, TreeView::ItemId)> on_selection_changed_;
};

TreeView::ItemId TreeView::Impl::g_next_id = 0;

TreeView::TreeView() : impl_(new TreeView::Impl()) {
    impl_->id_ = g_treeview_id++;
    impl_->root_.id = Impl::g_next_id++;
    impl_->id2item_[impl_->root_.id] = &impl_->root_;
}

TreeView::~TreeView() {}

TreeView::ItemId TreeView::GetRootItem() const { return impl_->root_.id; }

TreeView::ItemId TreeView::AddItem(ItemId parent_id, const char *text) {
    Impl::Item item;
    item.id = Impl::g_next_id++;
    // ImGUI uses the text to identify the item, create a ID string
    std::stringstream s;
    s << "treeview" << impl_->id_ << "item" << item.id;
    item.id_string = s.str();
    item.text = text;

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

void TreeView::RemoveItem(ItemId item_id) {
    auto item_it = impl_->id2item_.find(item_id);
    if (item_it != impl_->id2item_.end()) {
        auto *item = item_it->second;
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

const char *TreeView::GetItemText(ItemId item_id) const {
    auto item_it = impl_->id2item_.find(item_id);
    if (item_it != impl_->id2item_.end()) {
        return item_it->second->text.c_str();
    }
    return nullptr;
}

void TreeView::SetItemText(ItemId item_id, const char *text) {
    auto item_it = impl_->id2item_.find(item_id);
    if (item_it != impl_->id2item_.end()) {
        item_it->second->text = text;
    }
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
        std::function<void(const char *, ItemId)> on_selection_changed) {
    impl_->on_selection_changed_ = on_selection_changed;
}

Size TreeView::CalcPreferredSize(const Theme &theme) const {
    return Size(Widget::DIM_GROW, Widget::DIM_GROW);
}

Widget::DrawResult TreeView::Draw(const DrawContext &context) {
    auto &frame = GetFrame();

    DrawImGuiPushEnabledState();
    ImGui::SetCursorPosX(frame.x - context.uiOffsetX);
    ImGui::SetCursorPosY(frame.y - context.uiOffsetY);

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
    ImGui::BeginChild(impl_->id_, ImVec2(frame.width, frame.height), true);

    Impl::Item *new_selection = nullptr;

    std::function<void(Impl::Item &)> DrawItem;
    DrawItem = [&DrawItem, this, &frame, &context,
                &new_selection](Impl::Item &item) {
        // ImGUI's tree doesn't seem to support selected items,
        // so we have to draw our own selection.
        if (item.id == impl_->selected_id_) {
            auto h = ImGui::GetTextLineHeightWithSpacing();
            // Since we are in a child, the cursor is relative to the upper left
            // of the tree's frame. To draw directly to the window list we
            // need to the absolute coordinates (relative the OS window's
            // upper left)
            auto y = frame.y + ImGui::GetCursorPosY() - ImGui::GetScrollY();
            ImGui::GetWindowDrawList()->AddRectFilled(
                    ImVec2(frame.x, y), ImVec2(frame.GetRight(), y + h),
                    colorToImguiRGBA(context.theme.tree_selected_color));
        }

        int flags = ImGuiTreeNodeFlags_DefaultOpen;
        if (impl_->can_select_parents_) {
            flags |= ImGuiTreeNodeFlags_OpenOnDoubleClick;
            flags |= ImGuiTreeNodeFlags_OpenOnArrow;
        }
        if (item.children.empty()) {
            flags |= ImGuiTreeNodeFlags_Leaf;
        }
        bool is_selectable =
                (item.children.empty() || impl_->can_select_parents_);
        if (ImGui::TreeNodeEx(item.id_string.c_str(), flags, "%s",
                              item.text.c_str())) {
            if (ImGui::IsItemClicked() && is_selectable) {
                impl_->selected_id_ = item.id;
                new_selection = &item;
            }
            for (auto &child : item.children) {
                DrawItem(child);
            }
            ImGui::TreePop();
        } else {
            if (ImGui::IsItemClicked() && is_selectable) {
                impl_->selected_id_ = item.id;
                new_selection = &item;
            }
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
    auto result = Widget::DrawResult::NONE;
    if (new_selection) {
        if (impl_->on_selection_changed_) {
            impl_->on_selection_changed_(new_selection->text.c_str(),
                                         new_selection->id);
        }
        result = Widget::DrawResult::REDRAW;
    }

    return result;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
