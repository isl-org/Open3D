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

#include <Visualization/Visualizer/ViewControl.h>

namespace three {

class ViewControlWithEditing : public ViewControl
{
public:
	enum EditingMode {
		EDITING_FREEMODE = 0,
		EDITING_ORTHO_POSITIVE_X = 1,
		EDITING_ORTHO_NEGATIVE_X = 2,
		EDITING_ORTHO_POSITIVE_Y = 3,
		EDITING_ORTHO_NEGATIVE_Y = 4,
		EDITING_ORTHO_POSITIVE_Z = 5,
		EDITING_ORTHO_NEGATIVE_Z = 6,
	};

public:
	void Reset() override;
	void ChangeFieldOfView(double step) override;
	void Scale(double scale) override;
	void Rotate(double x, double y, double xo, double yo) override;
	void Translate(double x, double y, double xo, double yo) override;

	void SetEditingMode(EditingMode mode);
	std::string GetStatusString();

	void ToggleEditingX() {
		if (editing_mode_ == EDITING_ORTHO_POSITIVE_X) {
			SetEditingMode(EDITING_ORTHO_NEGATIVE_X);
		} else {
			SetEditingMode(EDITING_ORTHO_POSITIVE_X);
		}
	}

	void ToggleEditingY() {
		if (editing_mode_ == EDITING_ORTHO_POSITIVE_Y) {
			SetEditingMode(EDITING_ORTHO_NEGATIVE_Y);
		} else {
			SetEditingMode(EDITING_ORTHO_POSITIVE_Y);
		}
	}

	void ToggleEditingZ() {
		if (editing_mode_ == EDITING_ORTHO_POSITIVE_Z) {
			SetEditingMode(EDITING_ORTHO_NEGATIVE_Z);
		} else {
			SetEditingMode(EDITING_ORTHO_POSITIVE_Z);
		}
	}
	
	void ToggleLocking() { is_view_locked_ = !is_view_locked_; }
	
	bool IsLocked() const { return is_view_locked_; }

protected:
	EditingMode editing_mode_ = EDITING_FREEMODE;
	ViewParameters view_status_backup_;
	bool is_view_locked_ = false;
};

}	// namespace three
