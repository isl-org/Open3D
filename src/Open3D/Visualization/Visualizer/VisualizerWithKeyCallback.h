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

#include <map>

#include "Open3D/Visualization/Visualizer/Visualizer.h"

namespace open3d {
namespace visualization {

class VisualizerWithKeyCallback : public Visualizer {
public:
    typedef std::pair<int, std::function<bool(Visualizer *)>> KeyCallbackPair;

public:
    VisualizerWithKeyCallback();
    ~VisualizerWithKeyCallback() override;
    VisualizerWithKeyCallback(const VisualizerWithKeyCallback &) = delete;
    VisualizerWithKeyCallback &operator=(const VisualizerWithKeyCallback &) =
            delete;

public:
    void PrintVisualizerHelp() override;
    void RegisterKeyCallback(int key,
                             std::function<bool(Visualizer *)> callback);

protected:
    void KeyPressCallback(GLFWwindow *window,
                          int key,
                          int scancode,
                          int action,
                          int mods) override;
    std::string PrintKeyToString(int key);

protected:
    std::map<int, std::function<bool(Visualizer *)>> key_to_callback_;
};

}  // namespace visualization
}  // namespace open3d
