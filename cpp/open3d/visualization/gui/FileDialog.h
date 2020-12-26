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

#pragma once

#include <functional>

#include "open3d/visualization/gui/Dialog.h"

#define GUI_USE_NATIVE_FILE_DIALOG 1

namespace open3d {
namespace visualization {
namespace gui {

struct Theme;

class FileDialog : public Dialog {
    using Super = Dialog;

public:
    enum class Mode { OPEN, SAVE };

    FileDialog(Mode type, const char *title, const Theme &theme);
    virtual ~FileDialog();

    /// May either be a directory or a file. If path is a file, it will be
    /// selected if it exists. Defaults to current working directory if
    /// no path is specified.
    void SetPath(const char *path);

    /// 'filter' is a string of extensions separated by a space or comma.
    /// An empty filter string matches all extensions.
    ///    AddFilter(".jpg .png .gif", "Image file (.jpg, .png, .gif)")
    ///    AddFilter(".jpg", "JPEG image (.jpg)")
    ///    AddFilter("", "All files")
    void AddFilter(const char *filter, const char *description);

    /// The OnCancel and OnDone callbacks *must* be specified.
    void SetOnCancel(std::function<void()> on_cancel);
    /// The OnCancel and OnDone callbacks *must* be specified.
    void SetOnDone(std::function<void(const char *)> on_done);

    Size CalcPreferredSize(const Theme &theme) const override;

    void OnWillShow() override;

protected:
    void OnDone();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
