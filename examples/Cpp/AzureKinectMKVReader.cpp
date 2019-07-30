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


#include "Open3D/Open3D.h"

int main(int argc, char **argv) {
    using namespace open3d;
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    MKVReader mkv_reader;
    mkv_reader.Open("/home/dongw1/Workspace/kinect/data/test.mkv");

    auto json = mkv_reader.GetMetaData();
    for (auto iter = json.begin(); iter != json.end(); ++iter) {
        std::cout << iter.key() << " " << json[iter.name()] << "\n";
    }

    std::vector<unsigned long long > timestamps = {15462462346L, 412423, 124200, 0, 12400000};
    for (auto &ts : timestamps) {
        std::cout << "timestamp = " << ts << "ms\n";
        mkv_reader.SeekTimestamp(ts);
        auto rgbd = mkv_reader.NextFrame();
        if (rgbd) {
            auto color = std::make_shared<geometry::Image>(rgbd->color_);
            visualization::DrawGeometries({color});
        } else {
            utility::LogDebug("Null RGBD frame for timestamp {} (us)\n", ts);
        }
    }

    mkv_reader.Close();
}