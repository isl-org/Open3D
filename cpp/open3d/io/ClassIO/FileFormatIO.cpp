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

#include "FileFormatIO.h"

#include "open3d/utility/FileSystem.h"

#include <map>

namespace open3d {
namespace io {

static std::map<std::string, FileGeometry (*)(const std::string&)> gExt2Func = {
        {"glb", ReadFileGeometryTypeGLTF},
        {"gltf", ReadFileGeometryTypeGLTF},
        {"obj", ReadFileGeometryTypeOBJ},
        {"off", ReadFileGeometryTypeOFF},
        {"pcd", ReadFileGeometryTypePCD},
        {"ply", ReadFileGeometryTypePLY},
        {"pts", ReadFileGeometryTypePTS},
        {"stl", ReadFileGeometryTypeSTL},
        {"xyz", ReadFileGeometryTypeXYZ},
        {"xyzn", ReadFileGeometryTypeXYZN},
        {"xyzrgb", ReadFileGeometryTypeXYZRGB},
};

FileGeometry ReadFileGeometryType(const std::string& path) {
    auto ext = utility::filesystem::GetFileExtensionInLowerCase(path);
    auto it = gExt2Func.find(ext);
    if (it != gExt2Func.end()) {
        return it->second(path);
    } else {
        return CONTENTS_UNKNOWN;
    }
}

}  // namespace io
}  // namespace open3d
