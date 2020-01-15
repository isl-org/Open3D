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

#include "FileDialog.h"

#include "Button.h"
#include "Combobox.h"
#include "Label.h"
#include "Layout.h"
#include "ListView.h"
#include "TextEdit.h"
#include "Theme.h"
#include "Util.h"

#include "Open3D/Utility/Console.h"
#include "Open3D/Utility/Helper.h"
#include "Open3D/Utility/FileSystem.h"

#include <string>
#include <unordered_map>
#include <unordered_set>

// macOS sorts directories in with the files
// Windows and Linux (GTK) sort directories first.
#ifdef __APPLE__
#define INLINE_DIRS 1
#else
#define INLINE_DIRS 0
#endif // __APPLE__

namespace open3d {
namespace gui {

class DirEntry {
public:
    enum Type { DIR, FILE };

    DirEntry(const std::string& name, Type type) {
        type_ = type;
        name_ = name;
        if (type == DIR) {
            display_ = std::string("[ ] ") + name_ + "/";
        } else {
            display_ = std::string("    ") + name_;
        }
    }

    Type GetType() const { return type_; }
    const std::string& GetName() const { return name_; }
    const std::string& GetDisplayText() const { return display_; }

    bool operator==(const DirEntry& rhs) const {
        return (type_ == rhs.type_ && name_ == rhs.name_);
    }
    bool operator!=(const DirEntry& rhs) const { return !this->operator==(rhs); }

    bool operator<(const DirEntry& rhs) const {
#if INLINE_DIRS
        // Sort directories by name; if the OS allows directories and files
        // to have the same name, put directories first.
        if (name_ == rhs.name_) {
            if (type_ == rhs.type_) {
                return false;
            } else {
                return (type_ == DIR);
            }
        } else {
            return (name_ < rhs.name_);
        }
#else
        // Sort directories first, then files.
        // Within each category sort by name.
        if (type_ == rhs.type_) {
            return (name_ < rhs.name_);
        } else {
            return (type_ == DIR);
        }
#endif // INLINE_DIRS
    }

private:
    Type type_;
    std::string name_;
    std::string display_;
};

struct FileDialog::Impl {
    Type type;
    std::vector<DirEntry> entries;
    std::shared_ptr<TextEdit> filename;
    std::shared_ptr<Combobox> dirtree;
    std::shared_ptr<ListView> filelist;
    std::shared_ptr<Combobox> filter;
    std::unordered_map<int, std::unordered_set<std::string>> filterIdx2filter;
    std::shared_ptr<Horiz> filterRow;
    std::shared_ptr<Button> ok;
    std::shared_ptr<Button> cancel;
    std::function<void()> onCancel;
    std::function<void(const char *)> onDone;

    const DirEntry& GetSelectedEntry() {
        static DirEntry gBogus("", DirEntry::Type::FILE);

        int idx = this->filelist->GetSelectedIndex();
        if (idx >= 0) {
            return this->entries[idx];
        } else {
            return gBogus;
        }
    }

    void UpdateDirectoryListing() {
        auto path = CalcCurrentDirectory();

        std::vector<std::string> rawSubdirs, rawFiles;
        utility::filesystem::ListDirectory(path, rawSubdirs, rawFiles);

        this->entries.clear();
        this->entries.reserve(rawSubdirs.size() + rawFiles.size());
        for (auto &dir : rawSubdirs) {
            auto d = utility::filesystem::GetFileNameWithoutDirectory(dir);
            this->entries.emplace_back(d, DirEntry::Type::DIR);
        }
        std::unordered_set<std::string> filter;
        auto it = this->filterIdx2filter.find(this->filter->GetSelectedIndex());
        if (it != this->filterIdx2filter.end()) {
            filter = it->second;
        }
        for (auto &file : rawFiles) {
            auto f = utility::filesystem::GetFileNameWithoutDirectory(file);
            auto ext = utility::filesystem::GetFileExtensionInLowerCase(f);
            if (!ext.empty()) {
                ext = std::string(".") + ext;
            }
            if (filter.empty() || filter.find(ext) != filter.end()) {
                this->entries.emplace_back(f, DirEntry::Type::FILE);
            }
        }
        std::sort(this->entries.begin(), this->entries.end());

        // Include an entry for ".." for convenience on Linux.
        // Don't do this on macOS because the native dialog has neither
        // a back button nor a "..".  Non-technical users aren't going to
        // have any idea what ".." means, so its unclear if this should
        // go in Windows, too, or just Linux (which is pretty much all
        // technical people). Windows' file dialog does have some sense
        // of "previous directory", though, so maybe it's okay if we
        // include an up icon.
#ifndef __APPLE__
        if (path != "/") {
            this->entries.insert(this->entries.begin(),
                                 DirEntry("..", DirEntry::Type::DIR));
        }
#endif // __APPLE__

        std::vector<std::string> display;
        display.reserve(this->entries.size());
        for (auto &e : this->entries) {
            display.push_back(e.GetDisplayText());
        }

        this->filelist->SetSelectedIndex(-1);
        if (this->type == Type::OPEN) {
            this->filename->SetText("");
            UpdateOk();
        }
        this->filelist->SetItems(display);
    }

    std::string CalcCurrentDirectory() const {
        auto idx = this->dirtree->GetSelectedIndex();
        std::string path;
        for (int i = 0;  i <= idx;  ++i) {
            if (i >= 2) {  // 0 is "/", so don't need "/" until 2.
                path += "/";
            }
            path += dirtree->GetItem(i);
        }
        return path;
    }

    void UpdateOk() {
        this->ok->SetEnabled(std::string(this->filename->GetText()) != "");
    }
};

FileDialog::FileDialog(Type type, const char *title, const Theme& theme)
: Dialog("File")
, impl_(std::make_unique<FileDialog::Impl>()) {
    auto em = theme.fontSize;
    auto layout = std::make_shared<Vert>(0.5 * em, Margins(em));
    impl_->type = type;

    // 'filename' needs to always exist, as we use it to store the name of
    // the picked file, however, it is only displayed for SAVE.
    impl_->filename = std::make_shared<TextEdit>();
    if (type == Type::SAVE) {
        auto filenameLabel = std::make_shared<Label>("Save as:");
        auto horiz = std::make_shared<Horiz>();
        horiz->AddChild(Horiz::MakeStretch());
        horiz->AddChild(filenameLabel);
        horiz->AddChild(impl_->filename);
        horiz->AddChild(Horiz::MakeStretch());
        layout->AddChild(horiz);
    }

    impl_->dirtree = std::make_shared<Combobox>();
    layout->AddChild(Horiz::MakeCentered(impl_->dirtree));

    impl_->filelist = std::make_shared<ListView>();
    layout->AddChild(impl_->filelist);

    impl_->cancel = std::make_shared<Button>("Cancel");
    if (type == Type::OPEN) {
        impl_->ok = std::make_shared<Button>("Open");
    } else if (type == Type::SAVE) {
        impl_->ok = std::make_shared<Button>("Save");
    }

    impl_->filter = std::make_shared<Combobox>();
    auto filterLabel = std::make_shared<Label>("File type:");
    impl_->filterRow = std::make_shared<Horiz>();
    impl_->filterRow->AddChild(Horiz::MakeStretch());
    impl_->filterRow->AddChild(filterLabel);
    impl_->filterRow->AddChild(impl_->filter);
    impl_->filterRow->AddChild(Horiz::MakeStretch());
    impl_->filterRow->SetVisible(false);
    layout->AddChild(impl_->filterRow);

    auto horiz = std::make_shared<Horiz>(em);
    horiz->AddChild(Horiz::MakeStretch());
    horiz->AddChild(impl_->cancel);
    horiz->AddChild(impl_->ok);
    layout->AddChild(horiz);
    this->AddChild(layout);

    impl_->filename->SetOnTextChanged([this](const char*) {
        this->impl_->UpdateOk();
    });
    impl_->dirtree->SetOnValueChanged([this](const char*) {
        this->impl_->UpdateDirectoryListing();
    });
    impl_->filelist->SetOnValueChanged([this](const char* value,
                                              bool isDoubleClick) {
        auto &entry = this->impl_->GetSelectedEntry();
        if (isDoubleClick) {
            if (entry.GetType() == DirEntry::Type::FILE) {
                this->OnDone();
                return;
            } else {
                auto newDir = this->impl_->CalcCurrentDirectory();
                newDir = newDir + "/" + entry.GetName();
                this->SetPath(newDir.c_str());
            }
        } else {
            if (entry.GetType() ==  DirEntry::Type::FILE) {
                this->impl_->filename->SetText(entry.GetName().c_str());
            } else {
                if (this->impl_->type == Type::OPEN) {
                    this->impl_->filename->SetText("");
                }
            }
        }
        this->impl_->UpdateOk();
    });
    impl_->filter->SetOnValueChanged([this](const char*) {
        this->impl_->UpdateDirectoryListing();  // re-filter directory
    });
    impl_->cancel->SetOnClicked([this]() {
        if (this->impl_->onCancel) {
            this->impl_->onCancel();
        } else {
            utility::LogError("FileDialog: need to call SetOnClicked()");
        }
    });
    impl_->ok->SetOnClicked([this]() {
        this->OnDone();
    });

    SetPath(utility::filesystem::GetWorkingDirectory().c_str());

    impl_->UpdateOk();
}

FileDialog::~FileDialog() {
}

void FileDialog::SetPath(const char *path) {
    // Test cases:
    //   "../../test.abc", "/usr/lib/../local/bin",
    //   "/",              "c:/windows/system/winnt.dll"
    auto pathComponents = util::PathToComponents(path);
    if (pathComponents.empty()) {
        if (path[0] == '/' && path[1] == '\0') {
            // absolute path; the "/" component will be added
            // later, so don't do anything here
        } else {
            pathComponents.push_back(".");
        }
    }

    char firstChar = path[0]; // '/' doesn't get stored in components
    bool isRelative = (firstChar != '/');
    bool isWindowsPath = false;
    // Check for Windows full path (e.g. "d:")
    if (isRelative && pathComponents[0].size() >= 2
        && ((firstChar >= 'a' && firstChar <= 'z')
         || (firstChar >= 'A' && firstChar <= 'Z'))
        && pathComponents[0][1] == ':') {
        isRelative = false;
        isWindowsPath = true;
    }

    std::vector<std::string> components;
    if (!isWindowsPath) {
        components.push_back("/");
    }
    if (isRelative) {
        auto cwd = utility::filesystem::GetWorkingDirectory();
        auto cwdComponents = util::PathToComponents(cwd.c_str());
        components.insert(components.end(),
                          cwdComponents.begin(), cwdComponents.end());
    } else {
        // absolute path, don't need any prefix
    }

    for (auto &dir : pathComponents) {
        if (dir == ".") { /* ignore */
        } else if (dir == "..") {
            components.pop_back();
        } else {
            components.push_back(dir);
        }
    }

    std::string dirpath = "";
    for (auto &dir : components) {
        if (dirpath != "" && dirpath != "/") {
            dirpath += "/";
        }
        dirpath += dir;
    }
    bool isDir = utility::filesystem::DirectoryExists(dirpath);

    impl_->dirtree->ClearItems();
    size_t n = (isDir ? components.size() : components.size() - 1);
    for (size_t i = 0;  i < n;  ++i) {
        impl_->dirtree->AddItem(components[i].c_str());
    }
    impl_->dirtree->SetSelectedIndex(n - 1);
    impl_->UpdateDirectoryListing();

    if (!isDir) {
        impl_->filename->SetText(components.back().c_str());
    }
}

void FileDialog::AddFilter(const char *filter, const char *description) {
    std::vector<std::string> exts;
    utility::SplitString(exts, filter, ", ");

    std::unordered_set<std::string> extFilter;
    for (auto &ext : exts) {
        extFilter.insert(ext);
    }

    bool firstFilter = impl_->filterIdx2filter.empty();
    impl_->filterIdx2filter[impl_->filterIdx2filter.size()] = extFilter;
    impl_->filter->AddItem(description);
    if (firstFilter) {
        impl_->filter->SetSelectedIndex(0);
        impl_->UpdateDirectoryListing(); // apply filter
    }
    impl_->filterRow->SetVisible(true);
}

void FileDialog::SetOnCancel(std::function<void()> onCancel) {
    impl_->onCancel = onCancel;
}

void FileDialog::SetOnDone(std::function<void(const char *)> onDone) {
    impl_->onDone = onDone;
}

void FileDialog::OnDone() {
    if (this->impl_->onDone) {
        auto dir = this->impl_->CalcCurrentDirectory();
        auto name = this->impl_->GetSelectedEntry().GetName();
        this->impl_->onDone((dir + "/" + name).c_str());
    } else {
        utility::LogError("FileDialog: need to call SetOnDone()");
    }
}

Size FileDialog::CalcPreferredSize(const Theme &theme) const {
    auto em = theme.fontSize;
    auto width = std::max(25 * em, Super::CalcPreferredSize(theme).width);
    return Size(width, 30 * em);

}

} // namespace gui
} // namespace open3d
