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

#include <string>
#include <vector>

#include "open3d/visualization/gui/Native.h"

namespace open3d {
namespace visualization {
namespace gui {

struct FileDialog::Impl {
    FileDialog::Mode mode_;
    std::string path_;
    std::vector<std::pair<std::string, std::string>> filters_;
    std::function<void()> on_cancel_;
    std::function<void(const char *)> on_done_;
};

FileDialog::FileDialog(Mode mode, const char *title, const Theme &theme)
    : Dialog(title), impl_(new FileDialog::Impl()) {
    impl_->mode_ = mode;
}

FileDialog::~FileDialog() {}

void FileDialog::SetPath(const char *path) { impl_->path_ = path; }

void FileDialog::AddFilter(const char *filter, const char *description) {
    impl_->filters_.push_back(
            std::make_pair<std::string, std::string>(filter, description));
}

void FileDialog::SetOnCancel(std::function<void()> on_cancel) {
    impl_->on_cancel_ = on_cancel;
}

void FileDialog::SetOnDone(std::function<void(const char *)> on_done) {
    impl_->on_done_ = on_done;
}

Size FileDialog::CalcPreferredSize(const Theme &theme) const {
    return Size(0, 0);
}

void FileDialog::OnWillShow() {
    auto on_ok = [this](const char *path) { this->impl_->on_done_(path); };
    auto on_cancel = [this]() { this->impl_->on_cancel_(); };
    ShowNativeFileDialog(impl_->mode_, impl_->path_, impl_->filters_, on_ok,
                         on_cancel);
}

void FileDialog::OnDone() {}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d

#endif  // __APPLE__ && GUI_USE_NATIVE_FILE_DIALOG
