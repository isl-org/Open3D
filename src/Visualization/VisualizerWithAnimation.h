// ----------------------------------------------------------------------------
// -                       Open3DV: www.open3dv.org                           -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 Qianyi Zhou <Qianyi.Zhou@gmail.com>
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

#include "Visualizer.h"

namespace three {

class VisualizerWithAnimation : public Visualizer  {
public:
	VisualizerWithAnimation();
	virtual ~VisualizerWithAnimation();
	VisualizerWithAnimation(const VisualizerWithAnimation &) = delete;
	VisualizerWithAnimation &operator=(const VisualizerWithAnimation &) = 
			delete;

public:
	virtual void PrintVisualizerHelp();
	virtual void UpdateWindowTitle();
	void Play(bool recording = false);
	void RegisterRecordingFilenameFormat(std::string format) {
		recording_filename_format_ = format;
	}

protected:
	virtual bool InitViewControl();
	//virtual void WindowRefreshCallback(GLFWwindow *window);
	//virtual void WindowResizeCallback(GLFWwindow *window, int w, int h);
	virtual void MouseMoveCallback(GLFWwindow* window, double x, double y);
	virtual void MouseScrollCallback(GLFWwindow* window, double x, double y);
	virtual void MouseButtonCallback(GLFWwindow* window,
			int button, int action, int mods);
	virtual void KeyPressCallback(GLFWwindow *window,
			int key, int scancode, int action, int mods);
	//virtual void WindowCloseCallback(GLFWwindow *window);

protected:
	std::string recording_filename_format_ = "Animation_%06d.png";
	size_t recording_file_index_ = 0;
};

}	// namespace three
