// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FileFormatIO.h"

#include <map>

#include "open3d/utility/FileSystem.h"

namespace open3d {
namespace io {

static std::map<std::string, FileGeometry (*)(const std::string&)> gExt2Func = {
        {"glb", ReadFileGeometryTypeGLTF},
        {"gltf", ReadFileGeometryTypeGLTF},
        {"obj", ReadFileGeometryTypeOBJ},
        {"fbx", ReadFileGeometryTypeFBX},
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
