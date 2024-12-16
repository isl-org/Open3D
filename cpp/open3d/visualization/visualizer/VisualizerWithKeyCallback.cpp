// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/visualizer/VisualizerWithKeyCallback.h"

namespace open3d {

namespace visualization {

VisualizerWithKeyCallback::VisualizerWithKeyCallback() {}

VisualizerWithKeyCallback::~VisualizerWithKeyCallback() {}

void VisualizerWithKeyCallback::PrintVisualizerHelp() {
    Visualizer::PrintVisualizerHelp();
    utility::LogInfo("  -- Keys registered for callback functions --");
    utility::LogInfo("    ");
    for (auto &key_callback_pair : key_to_callback_) {
        utility::LogInfo("[{}] ", PrintKeyToString(key_callback_pair.first));
    }
    utility::LogInfo("");
    utility::LogInfo(
            "    The default functions of these keys will be overridden.");
    utility::LogInfo("");

    std::string mouse_callbacks = (mouse_move_callback_ ? "MouseMove, " : "");
    mouse_callbacks += (mouse_scroll_callback_ ? "MouseScroll, " : "");
    mouse_callbacks += (mouse_button_callback_ ? "MouseButton, " : "");
    utility::LogInfo("    Custom mouse callbacks registered for: {}",
                     mouse_callbacks.substr(0, mouse_callbacks.size() - 2));
    utility::LogInfo("");
}

void VisualizerWithKeyCallback::RegisterKeyCallback(
        int key, std::function<bool(Visualizer *)> callback) {
    key_to_callback_[key] = callback;
}

void VisualizerWithKeyCallback::RegisterKeyActionCallback(
        int key, std::function<bool(Visualizer *, int, int)> callback) {
    key_action_to_callback_[key] = callback;
}

void VisualizerWithKeyCallback::RegisterMouseMoveCallback(
        std::function<bool(Visualizer *, double, double)> callback) {
    mouse_move_callback_ = callback;
}

void VisualizerWithKeyCallback::RegisterMouseScrollCallback(
        std::function<bool(Visualizer *, double, double)> callback) {
    mouse_scroll_callback_ = callback;
}

void VisualizerWithKeyCallback::RegisterMouseButtonCallback(
        std::function<bool(Visualizer *, int, int, int)> callback) {
    mouse_button_callback_ = callback;
}

void VisualizerWithKeyCallback::KeyPressCallback(
        GLFWwindow *window, int key, int scancode, int action, int mods) {
    auto action_callback = key_action_to_callback_.find(key);
    if (action_callback != key_action_to_callback_.end()) {
        if (action_callback->second(this, action, mods)) {
            UpdateGeometry();
        }
        UpdateRender();
        return;
    }

    if (action == GLFW_RELEASE) {
        return;
    }
    auto callback = key_to_callback_.find(key);
    if (callback != key_to_callback_.end()) {
        if (callback->second(this)) {
            UpdateGeometry();
        }
        UpdateRender();
    } else {
        Visualizer::KeyPressCallback(window, key, scancode, action, mods);
    }
}

void VisualizerWithKeyCallback::MouseMoveCallback(GLFWwindow *window,
                                                  double x,
                                                  double y) {
    if (mouse_move_callback_) {
        if (mouse_move_callback_(this, x, y)) {
            UpdateGeometry();
        }
        UpdateRender();
    } else {
        Visualizer::MouseMoveCallback(window, x, y);
    }
}

void VisualizerWithKeyCallback::MouseScrollCallback(GLFWwindow *window,
                                                    double x,
                                                    double y) {
    if (mouse_scroll_callback_) {
        if (mouse_scroll_callback_(this, x, y)) {
            UpdateGeometry();
        }
        UpdateRender();
    } else {
        Visualizer::MouseScrollCallback(window, x, y);
    }
}

void VisualizerWithKeyCallback::MouseButtonCallback(GLFWwindow *window,
                                                    int button,
                                                    int action,
                                                    int mods) {
    if (mouse_button_callback_) {
        if (mouse_button_callback_(this, button, action, mods)) {
            UpdateGeometry();
        }
        UpdateRender();
    } else {
        Visualizer::MouseButtonCallback(window, button, action, mods);
    }
}

std::string VisualizerWithKeyCallback::PrintKeyToString(int key) {
    if (key == GLFW_KEY_SPACE) {  // 32
        return std::string("Space");
    } else if (key >= 39 && key <= 96) {  // 39 - 96
        return std::string(1, char(key));
    } else if (key == GLFW_KEY_ESCAPE) {  // 256
        return std::string("Esc");
    } else if (key == GLFW_KEY_ENTER) {  // 257
        return std::string("Enter");
    } else if (key == GLFW_KEY_TAB) {  // 258
        return std::string("Tab");
    } else if (key == GLFW_KEY_BACKSPACE) {  // 259
        return std::string("Backspace");
    } else if (key == GLFW_KEY_INSERT) {  // 260
        return std::string("Insert");
    } else if (key == GLFW_KEY_DELETE) {  // 261
        return std::string("Delete");
    } else if (key == GLFW_KEY_RIGHT) {  // 262
        return std::string("Right arrow");
    } else if (key == GLFW_KEY_LEFT) {  // 263
        return std::string("Left arrow");
    } else if (key == GLFW_KEY_DOWN) {  // 264
        return std::string("Down arrow");
    } else if (key == GLFW_KEY_UP) {  // 265
        return std::string("Up arrow");
    } else if (key == GLFW_KEY_PAGE_UP) {  // 266
        return std::string("Page up");
    } else if (key == GLFW_KEY_PAGE_DOWN) {  // 267
        return std::string("Page down");
    } else if (key == GLFW_KEY_HOME) {  // 268
        return std::string("Home");
    } else if (key == GLFW_KEY_END) {  // 269
        return std::string("End");
    } else if (key == GLFW_KEY_CAPS_LOCK) {  // 280
        return std::string("Caps lock");
    } else if (key == GLFW_KEY_SCROLL_LOCK) {  // 281
        return std::string("Scroll lock");
    } else if (key == GLFW_KEY_NUM_LOCK) {  // 282
        return std::string("Num lock");
    } else if (key == GLFW_KEY_PRINT_SCREEN) {  // 283
        return std::string("PrtScn");
    } else if (key == GLFW_KEY_PAUSE) {  // 284
        return std::string("Pause");
    } else if (key >= 290 && key <= 314) {  // 290 - 314
        return std::string("F") + std::to_string(key - 289);
    }
    return std::string("Unknown");
}

}  // namespace visualization
}  // namespace open3d
