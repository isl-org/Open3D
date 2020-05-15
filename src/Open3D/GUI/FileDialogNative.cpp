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

#if defined(__APPLE__)

#include "FileDialog.h"

#include "Native.h"

#include <string>
#include <vector>

namespace open3d {
namespace gui {

struct FileDialog::Impl {
    FileDialog::Mode mode;
    std::string path;
    std::vector<std::pair<std::string, std::string>> filters;
    std::function<void()> onCancel;
    std::function<void(const char *)> onDone;
};

FileDialog::FileDialog(Mode mode, const char *title, const Theme &theme)
    : Dialog(title), impl_(new FileDialog::Impl()) {
    impl_->mode = mode;
}

FileDialog::~FileDialog() {}

void FileDialog::SetPath(const char *path) { impl_->path = path; }

void FileDialog::AddFilter(const char *filter, const char *description) {
    impl_->filters.push_back(
            std::make_pair<std::string, std::string>(filter, description));
}

void FileDialog::SetOnCancel(std::function<void()> onCancel) {
    impl_->onCancel = onCancel;
}

void FileDialog::SetOnDone(std::function<void(const char *)> onDone) {
    impl_->onDone = onDone;
}

Size FileDialog::CalcPreferredSize(const Theme &theme) const {
    return Size(0, 0);
}

void FileDialog::OnWillShow() {
    auto onOk = [this](const char *path) { this->impl_->onDone(path); };
    auto onCancel = [this]() { this->impl_->onCancel(); };
    ShowNativeFileDialog(impl_->mode, impl_->path, impl_->filters, onOk,
                         onCancel);
}

void FileDialog::OnDone() {}

}  // namespace gui
}  // namespace open3d

#endif  // __APPLE__
