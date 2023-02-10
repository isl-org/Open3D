// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Logging.h"

#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(Logging, LogError) {
    EXPECT_THROW(utility::LogError("Example exception message."),
                 std::runtime_error);
}

TEST(Logging, LogInfo) {
    utility::LogInfo("{}", "Example shape print {1, 2, 3}.");
}

}  // namespace tests
}  // namespace open3d
