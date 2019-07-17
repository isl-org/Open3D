// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2019 www.open3d.org
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

#include <cstdio>

#include "Open3D/Open3D.h"

int main(int argc, char **argv) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    utility::LogDebug("This Debug message should be visible, {} {:.2f}\n",
                      "format:", 0.42001);
    utility::LogInfo("This Info message should be visible, {} {:.2f}\n",
                     "format:", 0.42001);
    utility::LogWarning("This Warning message should be visible, {} {:.2f}\n",
                        "format:", 0.42001);
    utility::LogError("This Error message should be visible, {} {:.2f}\n",
                      "format:", 0.42001);

    utility::LogDebugf("This Debug message should be visible, %s %.2f\n",
                       "formatf:", 0.42001);
    utility::LogInfof("This Info message should be visible, %s %.2f\n",
                      "formatf:", 0.42001);
    utility::LogWarningf("This Warning message should be visible, %s %.2f\n",
                         "formatf:", 0.42001);
    utility::LogErrorf("This Error message should be visible, %s %.2f\n",
                       "formatf:", 0.42001);

    utility::SetVerbosityLevel(utility::VerbosityLevel::Info);

    utility::LogDebug("This Debug message should NOT be visible, {} {:.2f}\n",
                      "format:", 0.42001);
    utility::LogInfo("This Info message should be visible, {} {:.2f}\n",
                     "format:", 0.42001);
    utility::LogWarning("This Warning message should be visible, {} {:.2f}\n",
                        "format:", 0.42001);
    utility::LogError("This Error message should be visible, {} {:.2f}\n",
                      "format:", 0.42001);

    utility::LogFatal("This Fatal message terminates the program\n");
    utility::LogFatal("This Fatal message should NOT be visible\n");

    return 0;
}
