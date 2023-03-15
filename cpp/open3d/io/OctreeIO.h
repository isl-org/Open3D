// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#include "open3d/geometry/Octree.h"

namespace open3d {
namespace io {

/// Factory function to create a octree from a file.
/// \return return an empty octree if fail to read the file.
std::shared_ptr<geometry::Octree> CreateOctreeFromFile(
        const std::string &filename, const std::string &format = "auto");

/// The general entrance for reading a Octree from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ReadOctree(const std::string &filename,
                geometry::Octree &octree,
                const std::string &format = "auto");

/// The general entrance for writing a Octree to a file
/// The function calls write functions based on the extension name of filename.
/// If the write function supports binary encoding and compression, the later
/// two parameters will be used. Otherwise they will be ignored.
/// \return return true if the write function is successful, false otherwise.
bool WriteOctree(const std::string &filename, const geometry::Octree &octree);

bool ReadOctreeFromJson(const std::string &filename, geometry::Octree &octree);

bool WriteOctreeToJson(const std::string &filename,
                       const geometry::Octree &octree);

}  // namespace io
}  // namespace open3d
