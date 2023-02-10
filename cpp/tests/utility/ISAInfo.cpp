// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/ISAInfo.h"

#include "open3d/utility/Logging.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(ISAInfo, GetSelectedISATarget) {
    EXPECT_NE(utility::ISAInfo::GetInstance().SelectedTarget(),
              utility::ISATarget::UNKNOWN);
}

}  // namespace tests
}  // namespace open3d
