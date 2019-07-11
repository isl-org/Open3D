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

#include "Open3D/Visualization/Visualizer/VisualizerWithKeyCallback.h"

namespace open3d {

namespace visualization {

VisualizerWithKeyCallback::VisualizerWithKeyCallback() {}

VisualizerWithKeyCallback::~VisualizerWithKeyCallback() {}

void VisualizerWithKeyCallback::PrintVisualizerHelp() {
    Visualizer::PrintVisualizerHelp();
    utility::LogInfo("  -- Keys registered for callback functions --\n");
    utility::LogInfo("    ");
    for (auto &key_callback_pair : key_to_callback_) {
        utility::LogInfo("[{}] ", PrintKeyToString(key_callback_pair.first));
    }
    utility::LogInfo("\n");
    utility::LogInfo(
            "    The default functions of these keys will be overridden.\n");
    utility::LogInfo("\n");
}

void VisualizerWithKeyCallback::RegisterKeyCallback(
        int key, std::function<bool(Visualizer *)> callback) {
    key_to_callback_[key] = callback;
}

void VisualizerWithKeyCallback::KeyPressCallback(
        GLFWwindow *window, int key, int scancode, int action, int mods) {
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
