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

#include "VisualizerForAlignment.h"

#include <External/tinyfiledialogs/tinyfiledialogs.h>

namespace three {

void VisualizerForAlignment::PrintVisualizerHelp()
{
	Visualizer::PrintVisualizerHelp();
	PrintInfo("  -- Alignment control --\n");
	PrintInfo("    Ctrl + S     : Save current alignment session into a JSON file.\n");
	PrintInfo("    Ctrl + O     : Load current alignment session from a JSON file.\n");
	PrintInfo("    Ctrl + A     : Align point clouds based on manually annotations.\n");
	PrintInfo("    Ctrl + R     : Run ICP refinement.\n");
	PrintInfo("    Ctrl + V     : Run voxel downsample for both source and target.");
	PrintInfo("    Ctrl + E     : Evaluate error and save to files.");
}

void VisualizerForAlignment::KeyPressCallback(GLFWwindow *window, int key,
		int scancode, int action, int mods)
{
	
}

}	// namespace three