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

#include "open3d/t/geometry/TSDFVoxelGrid.h"

namespace open3d {
namespace t {
namespace io {

/// Factory function to create a TSDFVoxelGrid from a file.
/// \return An empty TSDFVoxelGrid if fail to read the file.
std::shared_ptr<geometry::TSDFVoxelGrid> CreateTSDFVoxelGridFromFile(
        const std::string &filename);

/// The general entrance for reading a TSDFVoxelGrid from a file.
/// The entry is a json file, storing metadata and the path to npz for the
/// underlying volumetric hashmap.
/// \return True if reading is successful, false
/// otherwise.
bool ReadTSDFVoxelGrid(const std::string &filename,
                       geometry::TSDFVoxelGrid &tsdf_voxelgrid);

/// The general entrance for writing a TSDFVoxelGrid to a file.
/// The entry is a json file, storing metadata and the path to npz for the
/// underlying volumetric hashmap.
/// \return True if writing is successful, false otherwise.
bool WriteTSDFVoxelGrid(const std::string &filename,
                        const geometry::TSDFVoxelGrid &tsdf_voxelgrid);

}  // namespace io
}  // namespace t
}  // namespace open3d
