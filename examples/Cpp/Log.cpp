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

    utility::SetVerbosityLevel(utility::Logger::VerbosityLevel::Debug);

    utility::Timer timer;

    double printf_time = 0;
    timer.Start();
    for(int idx = 0; idx < 1000000; ++idx) {
        float f = idx / 3.14;
        int d = (idx * idx) % 97;
        printf("this is a message %s, %.6f, %d\n", "this is some string", f, d);
        fflush(stdout);
    }
    timer.Stop();
    printf_time += timer.GetDuration();

    double fmt_time = 0;
    timer.Start();
    for(int idx = 0; idx < 1000000; ++idx) {
        float f = idx / 3.14;
        int d = (idx * idx) % 97;
        utility::LogDebug("this is a message {}, {:.6}, {:d}\n", "this is some string", f, d);
        fflush(stdout);
    }
    timer.Stop();
    fmt_time += timer.GetDuration();

    double fmtf_time = 0;
    timer.Start();
    for(int idx = 0; idx < 1000000; ++idx) {
        float f = idx / 3.14;
        int d = (idx * idx) % 97;
        utility::LogDebugf("this is a message %s, %.6f, %d\n", "this is some string", f, d);
        fflush(stdout);
    }
    timer.Stop();
    fmtf_time += timer.GetDuration();

    utility::LogInfo("output using fmt took in total {}[ms]\n", fmt_time);
    utility::LogInfo("output using fmtf took in total {}[ms]\n", fmtf_time);
    utility::LogInfo("output using printf took in total {}[ms]\n", printf_time);

    return 0;
}
