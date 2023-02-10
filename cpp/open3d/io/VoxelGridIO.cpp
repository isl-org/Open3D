// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/VoxelGridIO.h"

#include <unordered_map>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {

namespace {
using namespace io;

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, geometry::VoxelGrid &, bool)>>
        file_extension_to_voxelgrid_read_function{
                {"ply", ReadVoxelGridFromPLY},
        };

static const std::unordered_map<std::string,
                                std::function<bool(const std::string &,
                                                   const geometry::VoxelGrid &,
                                                   const bool,
                                                   const bool,
                                                   const bool)>>
        file_extension_to_voxelgrid_write_function{
                {"ply", WriteVoxelGridToPLY},
        };
}  // unnamed namespace

namespace io {

std::shared_ptr<geometry::VoxelGrid> CreateVoxelGridFromFile(
        const std::string &filename,
        const std::string &format,
        bool print_progress) {
    auto voxelgrid = std::make_shared<geometry::VoxelGrid>();
    ReadVoxelGrid(filename, *voxelgrid, format, print_progress);
    return voxelgrid;
}

bool ReadVoxelGrid(const std::string &filename,
                   geometry::VoxelGrid &voxelgrid,
                   const std::string &format,
                   bool print_progress) {
    std::string filename_ext;
    if (format == "auto") {
        filename_ext =
                utility::filesystem::GetFileExtensionInLowerCase(filename);
    } else {
        filename_ext = format;
    }
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read geometry::VoxelGrid failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_voxelgrid_read_function.find(filename_ext);
    if (map_itr == file_extension_to_voxelgrid_read_function.end()) {
        utility::LogWarning(
                "Read geometry::VoxelGrid failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, voxelgrid, print_progress);
    utility::LogDebug("Read geometry::VoxelGrid: {:d} voxels.",
                      (int)voxelgrid.voxels_.size());
    return success;
}

bool WriteVoxelGrid(const std::string &filename,
                    const geometry::VoxelGrid &voxelgrid,
                    bool write_ascii /* = false*/,
                    bool compressed /* = false*/,
                    bool print_progress) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::VoxelGrid failed: unknown file extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_voxelgrid_write_function.find(filename_ext);
    if (map_itr == file_extension_to_voxelgrid_write_function.end()) {
        utility::LogWarning(
                "Write geometry::VoxelGrid failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, voxelgrid, write_ascii, compressed,
                                   print_progress);
    utility::LogDebug("Write geometry::VoxelGrid: {:d} voxels.",
                      (int)voxelgrid.voxels_.size());
    return success;
}

}  // namespace io
}  // namespace open3d
