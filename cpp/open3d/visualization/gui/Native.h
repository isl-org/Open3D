// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

#include "FileDialog.h"

struct GLFWwindow;

namespace open3d {
namespace visualization {
namespace gui {

void* GetNativeDrawable(GLFWwindow* glfw_window);
// Note that Windows cannot post an expose event so it must draw immediately.
// Therefore this function cannot be called while drawing.
void PostNativeExposeEvent(GLFWwindow* glfw_window);
void ShowNativeAlert(const char* message);

#ifdef __APPLE__
void MacTransformIntoApp();
void SetNativeMenubar(void* menubar);
#endif  // __APPLE_

#if defined(__APPLE__) || defined(_WIN32)
void ShowNativeFileDialog(
        FileDialog::Mode mode,
        const std::string& path,
        const std::vector<std::pair<std::string, std::string>>& filters,
        std::function<void(const char*)> on_ok,
        std::function<void()> on_cancel);
#endif  // __APPLE__ || _WIN32

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
