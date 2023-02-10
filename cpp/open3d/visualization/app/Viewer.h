// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <memory>

namespace open3d {
namespace visualization {
namespace app {

/// Runs Open3D Viewer. This function is called when the standalone viewer app
/// is run or when the draw command is called from the command line interface.

/// \param argc (argument count) is the number of arguments in \p argv .
/// \param argv (argument vector) is the array of arguments stored as character
/// arrays. It contains the path of the calling program (which should be in the
/// same directory as the gui resources folder) as the first argument. The
/// optional second argument is the path of the geometry file to be visualized
void RunViewer(int argc, const char *argv[]);

}  // namespace app
}  // namespace visualization
}  // namespace open3d
