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

namespace open3d {
namespace io {

enum FileGeometry {
    CONTENTS_UNKNOWN = 0,
    CONTAINS_POINTS = (1 << 0),
    CONTAINS_LINES = (1 << 1),
    CONTAINS_TRIANGLES = (1 << 2),
};

/// Returns the kind of geometry that the file contains. This is a quick
/// function designed to query the file in order to determine whether to
/// call ReadTriangleMesh(), ReadLineSet(), or ReadPointCloud()
FileGeometry ReadFileGeometryType(const std::string& path);

FileGeometry ReadFileGeometryTypeGLTF(const std::string& path);
FileGeometry ReadFileGeometryTypeOBJ(const std::string& path);
FileGeometry ReadFileGeometryTypeFBX(const std::string& path);
FileGeometry ReadFileGeometryTypeOFF(const std::string& path);
FileGeometry ReadFileGeometryTypePCD(const std::string& path);
FileGeometry ReadFileGeometryTypePLY(const std::string& path);
FileGeometry ReadFileGeometryTypePTS(const std::string& path);
FileGeometry ReadFileGeometryTypeSTL(const std::string& path);
FileGeometry ReadFileGeometryTypeXYZ(const std::string& path);
FileGeometry ReadFileGeometryTypeXYZN(const std::string& path);
FileGeometry ReadFileGeometryTypeXYZRGB(const std::string& path);

}  // namespace io
}  // namespace open3d
