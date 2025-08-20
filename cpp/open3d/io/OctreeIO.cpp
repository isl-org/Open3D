// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/OctreeIO.h"

#include <unordered_map>

#include "open3d/io/IJsonConvertibleIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace io {

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, geometry::Octree &)>>
        file_extension_to_octree_read_function{
                {"json", ReadOctreeFromJson},
                {"bin", ReadOctreeFromBIN},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, const geometry::Octree &)>>
        file_extension_to_octree_write_function{
                {"json", WriteOctreeToJson},
                {"bin", WriteOctreeToBIN},
        };

std::shared_ptr<geometry::Octree> CreateOctreeFromFile(
        const std::string &filename, const std::string &format) {
    auto octree = std::make_shared<geometry::Octree>();
    ReadOctree(filename, *octree);
    return octree;
}

bool ReadOctree(const std::string &filename,
                geometry::Octree &octree,
                const std::string &format) {
    std::string filename_ext;
    if (format == "auto") {
        filename_ext =
                utility::filesystem::GetFileExtensionInLowerCase(filename);
    } else {
        filename_ext = format;
    }
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read geometry::Octree failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_octree_read_function.find(filename_ext);
    if (map_itr == file_extension_to_octree_read_function.end()) {
        utility::LogWarning(
                "Read geometry::Octree failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, octree);
    utility::LogDebug("Read geometry::Octree.");
    return success;
}

bool WriteOctree(const std::string &filename, const geometry::Octree &octree) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::Octree failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_octree_write_function.find(filename_ext);
    if (map_itr == file_extension_to_octree_write_function.end()) {
        utility::LogWarning(
                "Write geometry::Octree failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, octree);
    utility::LogDebug("Write geometry::Octree.");
    return success;
}

bool ReadOctreeFromJson(const std::string &filename, geometry::Octree &octree) {
    return ReadIJsonConvertible(filename, octree);
}

bool WriteOctreeToJson(const std::string &filename,
                       const geometry::Octree &octree) {
    return WriteIJsonConvertibleToJSON(filename, octree);
}

bool ReadOctreeFromBIN(const std::string &filename, geometry::Octree &octree) {
    std::string bin_data;
    if (!ReadOctreeBinaryStreamFromBIN(filename, bin_data)) {
        return false;
    }
    // Deserialize bin_data into octree
    octree.DeserializeFromBinaryStream(bin_data);
    return true;
}

bool WriteOctreeToBIN(const std::string &filename,
                      const geometry::Octree &octree) {
    std::string bin_data;
    octree.SerializeToBinaryStream(bin_data);
    return WriteOctreeBinaryStreamToBIN(filename, bin_data);
}
}  // namespace io
}  // namespace open3d
