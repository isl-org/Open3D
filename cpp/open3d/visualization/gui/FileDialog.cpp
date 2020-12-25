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

// Include FileDialog here to get value of GUI_USE_NATIVE_FILE_DIALOG
#include "open3d/visualization/gui/FileDialog.h"

#if defined(__APPLE__) && GUI_USE_NATIVE_FILE_DIALOG
// see FileDialogNative.cpp
#else

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "open3d/utility/Console.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Helper.h"
#include "open3d/visualization/gui/Button.h"
#include "open3d/visualization/gui/Combobox.h"
#include "open3d/visualization/gui/Label.h"
#include "open3d/visualization/gui/Layout.h"
#include "open3d/visualization/gui/ListView.h"
#include "open3d/visualization/gui/TextEdit.h"
#include "open3d/visualization/gui/Theme.h"
#include "open3d/visualization/gui/Util.h"

// macOS sorts directories in with the files
// Windows and Linux (GTK) sort directories first.
#ifdef __APPLE__
#define INLINE_DIRS 1
#else
#define INLINE_DIRS 0
#endif  // __APPLE__

namespace open3d {
namespace visualization {
namespace gui {

namespace {
// The current path of the dialog should persist across runs of the dialog.
// This is defined here rather than in the class definition because we don't
// need to be exporting internal details of the class.
static std::string g_file_dialog_dir;
}  // namespace

class DirEntry {
public:
    enum Type { DIR, FILE };

    DirEntry(const std::string &name, Type type) {
        type_ = type;
        name_ = name;
        if (type == DIR) {
            display_ = std::string("[ ] ") + name_ + "/";
        } else {
            display_ = std::string("    ") + name_;
        }
    }

    Type GetType() const { return type_; }
    const std::string &GetName() const { return name_; }
    const std::string &GetDisplayText() const { return display_; }

    bool operator==(const DirEntry &rhs) const {
        return (type_ == rhs.type_ && name_ == rhs.name_);
    }
    bool operator!=(const DirEntry &rhs) const {
        return !this->operator==(rhs);
    }

    bool operator<(const DirEntry &rhs) const {
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
#endif  // INLINE_DIRS
    }

private:
    Type type_;
    std::string name_;
    std::string display_;
};

struct FileDialog::Impl {
    Mode mode_;
    std::vector<DirEntry> entries_;
    std::shared_ptr<TextEdit> filename_;
    std::shared_ptr<Combobox> dirtree_;
    std::shared_ptr<ListView> filelist_;
    std::shared_ptr<Combobox> filter_;
    std::unordered_map<int, std::unordered_set<std::string>>
            filter_idx_2_filter;
    std::shared_ptr<Horiz> filter_row_;
    std::shared_ptr<Button> ok_;
    std::shared_ptr<Button> cancel_;
    std::function<void()> on_cancel_;
    std::function<void(const char *)> on_done_;

    const DirEntry &GetSelectedEntry() {
        static DirEntry g_bogus("", DirEntry::Type::FILE);

        int idx = filelist_->GetSelectedIndex();
        if (idx >= 0) {
            return entries_[idx];
        } else {
            return g_bogus;
        }
    }

    void UpdateDirectoryListing() {
        auto path = CalcCurrentDirectory();

        std::vector<std::string> raw_subdirs, raw_files;
        utility::filesystem::ListDirectory(path, raw_subdirs, raw_files);

        entries_.clear();
        entries_.reserve(raw_subdirs.size() + raw_files.size());
        for (auto &dir : raw_subdirs) {
            auto d = utility::filesystem::GetFileNameWithoutDirectory(dir);
            entries_.emplace_back(d, DirEntry::Type::DIR);
        }
        std::unordered_set<std::string> filter;
        auto it = filter_idx_2_filter.find(filter_->GetSelectedIndex());
        if (it != filter_idx_2_filter.end()) {
            filter = it->second;
        }
        for (auto &file : raw_files) {
            auto f = utility::filesystem::GetFileNameWithoutDirectory(file);
            auto ext = utility::filesystem::GetFileExtensionInLowerCase(f);
            if (!ext.empty()) {
                ext = std::string(".") + ext;
            }
            if (filter.empty() || filter.find(ext) != filter.end()) {
                entries_.emplace_back(f, DirEntry::Type::FILE);
            }
        }
        std::sort(entries_.begin(), entries_.end());

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
            entries_.insert(entries_.begin(),
                            DirEntry("..", DirEntry::Type::DIR));
        }
#endif  // __APPLE__

        std::vector<std::string> display;
        display.reserve(entries_.size());
        for (auto &e : entries_) {
            display.push_back(e.GetDisplayText());
        }

        filelist_->SetSelectedIndex(-1);
        if (mode_ == Mode::OPEN) {
            filename_->SetText("");
            UpdateOk();
        }
        filelist_->SetItems(display);
    }

    std::string CalcCurrentDirectory() const {
#ifdef _WIN32
        const int nSkipSlash = 1;
#else
        const int nSkipSlash = 2;  // 0 is "/", so don't need "/" until 2.
#endif  // _WIN32
        auto idx = dirtree_->GetSelectedIndex();
        std::string path;
        for (int i = 0; i <= idx; ++i) {
            if (i >= nSkipSlash) {
                path += "/";
            }
            path += dirtree_->GetItem(i);
        }
        return path;
    }

    void UpdateOk() {
        ok_->SetEnabled(std::string(filename_->GetText()) != "");
    }
};

FileDialog::FileDialog(Mode mode, const char *title, const Theme &theme)
    : Dialog("File"), impl_(new FileDialog::Impl()) {
    auto em = theme.font_size;
    auto layout = std::make_shared<Vert>(int(std::ceil(0.5 * em)), Margins(em));
    impl_->mode_ = mode;

    // 'filename' needs to always exist, as we use it to store the name of
    // the picked file, however, it is only displayed for SAVE.
    impl_->filename_ = std::make_shared<TextEdit>();
    if (mode == Mode::SAVE) {
        auto filenameLabel = std::make_shared<Label>("Save as:");
        auto horiz = std::make_shared<Horiz>();
        horiz->AddStretch();
        horiz->AddChild(filenameLabel);
        horiz->AddChild(impl_->filename_);
        horiz->AddStretch();
        layout->AddChild(horiz);
    }

    impl_->dirtree_ = std::make_shared<Combobox>();
    layout->AddChild(Horiz::MakeCentered(impl_->dirtree_));

    impl_->filelist_ = std::make_shared<ListView>();
    layout->AddChild(impl_->filelist_);

    impl_->cancel_ = std::make_shared<Button>("Cancel");
    if (mode == Mode::OPEN) {
        impl_->ok_ = std::make_shared<Button>("Open");
    } else if (mode == Mode::SAVE) {
        impl_->ok_ = std::make_shared<Button>("Save");
    }

    impl_->filter_ = std::make_shared<Combobox>();
    auto filter_label = std::make_shared<Label>("File type:");
    impl_->filter_row_ = std::make_shared<Horiz>();
    impl_->filter_row_->AddStretch();
    impl_->filter_row_->AddChild(filter_label);
    impl_->filter_row_->AddChild(impl_->filter_);
    impl_->filter_row_->AddStretch();
    impl_->filter_row_->SetVisible(false);
    layout->AddChild(impl_->filter_row_);

    auto horiz = std::make_shared<Horiz>(em);
    horiz->AddStretch();
    horiz->AddChild(impl_->cancel_);
    horiz->AddChild(impl_->ok_);
    layout->AddChild(horiz);
    this->AddChild(layout);

    impl_->filename_->SetOnTextChanged(
            [this](const char *) { this->impl_->UpdateOk(); });
    impl_->dirtree_->SetOnValueChanged([this](const char *, int) {
        this->impl_->UpdateDirectoryListing();
    });
    impl_->filelist_->SetOnValueChanged([this](const char *value,
                                               bool is_double_click) {
        auto &entry = this->impl_->GetSelectedEntry();
        if (is_double_click) {
            if (entry.GetType() == DirEntry::Type::FILE) {
                this->OnDone();
                return;
            } else {
                auto new_dir = this->impl_->CalcCurrentDirectory();
                new_dir = new_dir + "/" + entry.GetName();
                this->SetPath(new_dir.c_str());
            }
        } else {
            if (entry.GetType() == DirEntry::Type::FILE) {
                this->impl_->filename_->SetText(entry.GetName().c_str());
            } else {
                if (this->impl_->mode_ == Mode::OPEN) {
                    this->impl_->filename_->SetText("");
                }
            }
        }
        this->impl_->UpdateOk();
    });
    impl_->filter_->SetOnValueChanged([this](const char *, int) {
        this->impl_->UpdateDirectoryListing();  // re-filter directory
    });
    impl_->cancel_->SetOnClicked([this]() {
        if (this->impl_->on_cancel_) {
            this->impl_->on_cancel_();
        } else {
            utility::LogError("FileDialog: need to call SetOnClicked()");
        }
    });
    impl_->ok_->SetOnClicked([this]() { this->OnDone(); });

    if (g_file_dialog_dir == "") {
        g_file_dialog_dir = utility::filesystem::GetWorkingDirectory();
    }
    SetPath(g_file_dialog_dir.c_str());

    impl_->UpdateOk();
}

FileDialog::~FileDialog() {}

void FileDialog::SetPath(const char *path) {
    auto components = utility::filesystem::GetPathComponents(path);

    std::string dirpath = "";
    for (auto &dir : components) {
        if (dirpath != "" && dirpath != "/") {
            dirpath += "/";
        }
        dirpath += dir;
    }
    bool is_dir = utility::filesystem::DirectoryExists(dirpath);

    impl_->dirtree_->ClearItems();
    int n = int(is_dir ? components.size() : components.size() - 1);
    for (int i = 0; i < n; ++i) {
        impl_->dirtree_->AddItem(components[i].c_str());
    }
    impl_->dirtree_->SetSelectedIndex(n - 1);
    impl_->UpdateDirectoryListing();
    if (is_dir) {
        g_file_dialog_dir = dirpath;
    }

    if (!is_dir) {
        impl_->filename_->SetText(components.back().c_str());
    }
}

void FileDialog::AddFilter(const char *filter, const char *description) {
    std::vector<std::string> exts;
    utility::SplitString(exts, filter, ", ");

    std::unordered_set<std::string> ext_filter;
    for (auto &ext : exts) {
        ext_filter.insert(ext);
    }

    bool first_filter = impl_->filter_idx_2_filter.empty();
    impl_->filter_idx_2_filter[int(impl_->filter_idx_2_filter.size())] =
            ext_filter;
    impl_->filter_->AddItem(description);
    if (first_filter) {
        impl_->filter_->SetSelectedIndex(0);
        impl_->UpdateDirectoryListing();  // apply filter
    }
    impl_->filter_row_->SetVisible(true);
}

void FileDialog::SetOnCancel(std::function<void()> on_cancel) {
    impl_->on_cancel_ = on_cancel;
}

void FileDialog::SetOnDone(std::function<void(const char *)> on_done) {
    impl_->on_done_ = on_done;
}

void FileDialog::OnWillShow() {}

void FileDialog::OnDone() {
    if (this->impl_->on_done_) {
        auto dir = this->impl_->CalcCurrentDirectory();
        utility::filesystem::ChangeWorkingDirectory(dir);
        std::string name = this->impl_->filename_->GetText();
        // If the user didn't specify an extension, automatically add one
        // (unless we don't have the any-files (*.*) filter selected).
        if (name.find(".") == std::string::npos && !name.empty()) {
            int idx = this->impl_->filter_->GetSelectedIndex();
            if (idx >= 0) {
                auto &exts = impl_->filter_idx_2_filter[idx];
                // Prefer PNG if available (otherwise in a list of common
                // image files, e.g., ".jpg .png", we might pick the lossy one.
                if (exts.find(".png") != exts.end()) {
                    name += ".png";
                } else {
                    if (!exts.empty()) {
                        name += *exts.begin();
                    }
                }
            }
        }
        std::cout << "[o3d] name: '" << name << "'" << std::endl;
        this->impl_->on_done_((dir + "/" + name).c_str());
    } else {
        utility::LogError("FileDialog: need to call SetOnDone()");
    }
}

Size FileDialog::CalcPreferredSize(const Theme &theme) const {
    auto em = theme.font_size;
    auto width = std::max(25 * em, Super::CalcPreferredSize(theme).width);
    return Size(width, 30 * em);
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d

#endif  // __APPLE__
