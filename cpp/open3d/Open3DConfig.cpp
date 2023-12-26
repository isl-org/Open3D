// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/Open3DConfig.h"

#include "open3d/utility/Logging.h"

namespace open3d {

void PrintOpen3DVersion() { utility::LogInfo("Open3D {}", OPEN3D_VERSION); }

}  // namespace open3d
