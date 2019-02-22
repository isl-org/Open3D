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

#include "VoxelGridIO.h"

#include <unordered_map>
#include <Core/Utility/Console.h>
#include <Core/Utility/FileSystem.h>

namespace open3d {

namespace {

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, VoxelGrid &)>>
        file_extension_to_voxelgrid_read_function{
                {"ply", ReadVoxelGridFromPLY},
        };

static const std::unordered_map<std::string,
                                std::function<bool(const std::string &,
                                                   const VoxelGrid &,
                                                   const bool,
                                                   const bool)>>
        file_extension_to_voxelgrid_write_function{
                {"ply", WriteVoxelGridToPLY},
        };
}  // unnamed namespace

std::shared_ptr<VoxelGrid> CreateVoxelGridFromFile(const std::string &filename,
                                                   const std::string &format) {
    auto voxelgrid = std::make_shared<VoxelGrid>();
    ReadVoxelGrid(filename, *voxelgrid, format);
    return voxelgrid;
}

bool ReadVoxelGrid(const std::string &filename,
                   VoxelGrid &voxelgrid,
                   const std::string &format) {
    std::string filename_ext;
    if (format == "auto") {
        filename_ext = filesystem::GetFileExtensionInLowerCase(filename);
    } else {
        filename_ext = format;
    }
    if (filename_ext.empty()) {
        PrintWarning("Read VoxelGrid failed: unknown file extension.\n");
        return false;
    }
    auto map_itr = file_extension_to_voxelgrid_read_function.find(filename_ext);
    if (map_itr == file_extension_to_voxelgrid_read_function.end()) {
        PrintWarning("Read VoxelGrid failed: unknown file extension.\n");
        return false;
    }
    bool success = map_itr->second(filename, voxelgrid);
    PrintDebug("Read VoxelGrid: %d voxels.\n", (int)voxelgrid.voxels_.size());
    return success;
}

bool WriteVoxelGrid(const std::string &filename,
                    const VoxelGrid &voxelgrid,
                    bool write_ascii /* = false*/,
                    bool compressed /* = false*/) {
    std::string filename_ext =
            filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        PrintWarning("Write VoxelGrid failed: unknown file extension.\n");
        return false;
    }
    auto map_itr =
            file_extension_to_voxelgrid_write_function.find(filename_ext);
    if (map_itr == file_extension_to_voxelgrid_write_function.end()) {
        PrintWarning("Write VoxelGrid failed: unknown file extension.\n");
        return false;
    }
    bool success =
            map_itr->second(filename, voxelgrid, write_ascii, compressed);
    PrintDebug("Write VoxelGrid: %d voxels.\n", (int)voxelgrid.voxels_.size());
    return success;
}

}  // namespace open3d
