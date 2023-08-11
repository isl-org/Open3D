// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
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
    enum class Mode { OPEN, SAVE, OPEN_DIR };

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

    Size CalcPreferredSize(const LayoutContext &context,
                           const Constraints &constraints) const override;

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
