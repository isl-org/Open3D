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

#include <string>

#include "open3d/io/PointCloudIO.h"
#include "open3d/t/geometry/PointCloud.h"

namespace open3d {
namespace io {

/// Factory function to create a pointcloud from a file
/// Return an empty pointcloud if fail to read the file.
std::shared_ptr<t::geometry::PointCloud> CreatetPointCloudFromFile(
        const std::string &filename,
        const std::string &format = "auto",
        bool print_progress = false);

/// The general entrance for reading a PointCloud from a file
/// The function calls read functions based on the extension name of filename.
/// See \p ReadPointCloudOption for additional options you can pass.
/// \return return true if the read function is successful, false otherwise.
bool ReadPointCloud(const std::string &filename,
                    t::geometry::PointCloud &pointcloud,
                    const ReadPointCloudOption &params = {});

/// The general entrance for writing a PointCloud to a file
/// The function calls write functions based on the extension name of filename.
/// See \p WritePointCloudOption for additional options you can pass.
/// \return return true if the write function is successful, false otherwise.
bool WritePointCloud(const std::string &filename,
                     const t::geometry::PointCloud &pointcloud,
                     const WritePointCloudOption &params = {});

bool ReadPointCloudFromXYZI(const std::string &filename,
                            t::geometry::PointCloud &pointcloud,
                            const ReadPointCloudOption &params);

bool WritePointCloudToXYZI(const std::string &filename,
                           const t::geometry::PointCloud &pointcloud,
                           const WritePointCloudOption &params);

}  // namespace io
}  // namespace open3d
