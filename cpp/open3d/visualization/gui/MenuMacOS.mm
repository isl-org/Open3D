// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2023 www.open3d.org
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

#include "open3d/visualization/gui/MenuMacOS.h"

#if __APPLE__

#import "AppKit/AppKit.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "open3d/visualization/gui/Application.h"

@interface Open3DRunnable : NSObject
{
    std::function<void()> action_;
}
@end

@implementation Open3DRunnable
- (id)initWithFunction: (std::function<void()>)f {
    self = [super init];
    if (self) {
        action_ = f;
    }
    return self;
}

- (void)run {
    action_();
}
@end

namespace open3d {
namespace visualization {
namespace gui {

struct MenuMacOS::Impl {
    NSMenu *menu_;
    std::vector<std::shared_ptr<MenuMacOS>> submenus_;

    NSMenuItem* FindMenuItem(ItemId itemId) const {

        auto item = [this->menu_ itemWithTag:itemId];
        if (item != nil) {
            return item;
        }
        // Not sure if -itemWithTag searches recursively
        for (auto sm : this->submenus_) {
            item = sm->impl_->FindMenuItem(itemId);
            if (item != nil) {
                return item;
            }
        }

        return nil;
    }

};

MenuMacOS::MenuMacOS()
: impl_(new MenuMacOS::Impl()) {
    impl_->menu_ = [[NSMenu alloc] initWithTitle:@""];
    impl_->menu_.autoenablesItems = NO;
}

MenuMacOS::~MenuMacOS() {} // ARC will automatically release impl_->menu

void* MenuMacOS::GetNativePointer() { return impl_->menu_; }

void MenuMacOS::AddItem(const char *name,
                        ItemId item_id /*= NO_ITEM*/,
                        KeyName key /*= KEY_NONE*/) {
    InsertItem(impl_->menu_.numberOfItems, name, item_id, key);
}

void MenuMacOS::AddMenu(const char *name, std::shared_ptr<MenuBase> submenu) {
    InsertMenu(impl_->menu_.numberOfItems, name, submenu);
}

void MenuMacOS::AddSeparator() {
    [impl_->menu_ addItem: [NSMenuItem separatorItem]];
}

void MenuMacOS::InsertItem(int index,
                           const char* name,
                           ItemId item_id /*= NO_ITEM*/,
                           KeyName key /*= KEY_NONE*/) {
    std::string shortcut;
    shortcut += char(key);
    NSString *objc_shortcut = [NSString stringWithUTF8String:shortcut.c_str()];
    auto item = [[NSMenuItem alloc]
                 initWithTitle:[NSString stringWithUTF8String:name]
                        action:@selector(run)
                 keyEquivalent:objc_shortcut];
    item.target = [[Open3DRunnable alloc] initWithFunction:[item_id]() {
        Application::GetInstance().OnMenuItemSelected(item_id);
    }];
    item.tag = item_id;
    if (index < impl_->menu_.numberOfItems) {
        [impl_->menu_ insertItem:item atIndex: index];
    } else {
        [impl_->menu_ addItem:item];
    }
}

void MenuMacOS::InsertMenu(int index, const char* name,
                           std::shared_ptr<MenuBase> submenu) {
    auto macos_submenu = std::dynamic_pointer_cast<MenuMacOS>(submenu);
    macos_submenu->impl_->menu_.title = [NSString stringWithUTF8String:name];
    auto item = [[NSMenuItem alloc]
                 initWithTitle:[NSString stringWithUTF8String:name]
                        action:nil
                 keyEquivalent:@""];
    if (index < impl_->menu_.numberOfItems) {
        [impl_->menu_ insertItem:item atIndex: index];
    } else {
        [impl_->menu_ addItem:item];
    }
    [impl_->menu_ setSubmenu:macos_submenu->impl_->menu_ forItem:item];
    impl_->submenus_.insert(impl_->submenus_.begin() + index, macos_submenu);
}

void MenuMacOS::InsertSeparator(int index) {
    [impl_->menu_ insertItem: [NSMenuItem separatorItem] atIndex: index];
}

int MenuMacOS::GetNumberOfItems() const {
    return impl_->menu_.numberOfItems;
}

bool MenuMacOS::IsEnabled(ItemId item_id) const {
    NSMenuItem *item = impl_->FindMenuItem(item_id);
    if (item) {
        return (item.enabled == YES ? true : false);
    }
    return false;
}

void MenuMacOS::SetEnabled(ItemId item_id, bool enabled) {
    NSMenuItem *item = impl_->FindMenuItem(item_id);
    if (item) {
        item.enabled = (enabled ? YES : NO);
    }
}

bool MenuMacOS::IsChecked(ItemId item_id) const {
    NSMenuItem *item = impl_->FindMenuItem(item_id);
    if (item) {
        return (item.state == NSControlStateValueOn);
    }
    return false;
}

void MenuMacOS::SetChecked(ItemId item_id, bool checked) {
    NSMenuItem *item = impl_->FindMenuItem(item_id);
    if (item) {
        item.state = (checked ? NSControlStateValueOn
                              : NSControlStateValueOff);
    }
}

int MenuMacOS::CalcHeight(const Theme &theme) const {
    return 0;  // menu is not part of window on macOS
}

bool MenuMacOS::CheckVisibilityChange() const {
    return false;
}

MenuBase::ItemId MenuMacOS::DrawMenuBar(const DrawContext &context,
                                        bool is_enabled) {
    return NO_ITEM;
}

MenuBase::ItemId MenuMacOS::Draw(const DrawContext &context,
                                 const char *name,
                                 bool is_enabled) {
    return NO_ITEM;
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d

#endif  // __APPLE__
