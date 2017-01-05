// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
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

#include <Visualization/Visualizer/Visualizer.h>

namespace three {

class SelectionPolygon;
class PointCloudPicker;

class VisualizerWithEditing : public Visualizer
{
public:
	enum SelectionMode {
		SELECTION_NONE = 0,
		SELECTION_RECTANGLE = 1,
		SELECTION_POLYGON = 2,
	};
	
public:
	VisualizerWithEditing();
	~VisualizerWithEditing() override;
	VisualizerWithEditing(const VisualizerWithEditing &) = delete;
	VisualizerWithEditing &operator=(const VisualizerWithEditing &) = delete;

public:
	void PrintVisualizerHelp() override;
	void UpdateWindowTitle() override;
	void BuildUtilities() override;
	int PickPoint(double x, double y, size_t geometry_index = 0);
	std::vector<size_t> &GetPickedPoints();

protected:
	bool InitViewControl() override;
	bool InitRenderOption() override;
	void WindowResizeCallback(GLFWwindow *window, int w, int h) override;
	void MouseMoveCallback(GLFWwindow* window, double x, double y) override;
	void MouseScrollCallback(GLFWwindow* window, double x, double y) override;
	void MouseButtonCallback(GLFWwindow* window,
			int button, int action, int mods) override;
	void KeyPressCallback(GLFWwindow *window,
			int key, int scancode, int action, int mods) override;
	void InvalidateSelectionPolygon();
	void SaveCroppingResult(const std::string &filename = "",
			int geometry_index = 0);

protected:
	std::shared_ptr<SelectionPolygon> selection_polygon_ptr_;
	std::shared_ptr<glsl::SelectionPolygonRenderer>
			selection_polygon_renderer_ptr_;
	SelectionMode selection_mode_ = SELECTION_NONE;
	
	std::shared_ptr<PointCloudPicker> pointcloud_picker_ptr_;
	std::shared_ptr<glsl::PointCloudPickerRenderer>
			pointcloud_picker_renderer_ptr_;
};

}	// namespace three
