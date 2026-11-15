// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Logging.h"

#include <string>

#include "tests/Tests.h"

namespace {

// A dummy class declared after including "open3d/utility/Logging.h" that
// defines member functions with the same names as Open3D's logging functions.
// If LogError/LogWarning/LogInfo/LogDebug were defined as unprefixed macros in
// the global preprocessor scope, the preprocessor would rewrite these member
// declarations and call sites into invalid `Logger::Log*_(...)` expressions.
struct UnrelatedLogger {
    int LogError() const { return 1; }
    int LogWarning() const { return 2; }
    int LogInfo() const { return 3; }
    int LogDebug() const { return 4; }
};

}  // namespace

namespace open3d {
namespace tests {

TEST(Logging, LogError) {
    EXPECT_THROW(utility::LogError("Example exception message."),
                 std::runtime_error);
    try {
        utility::LogError("Formatted error: {}", 42);
        FAIL() << "Expected std::runtime_error";
    } catch (const std::runtime_error &e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("Formatted error: 42"), std::string::npos);
        EXPECT_NE(msg.find("Logging.cpp"), std::string::npos);
    }
}

TEST(Logging, LogInfo) {
    utility::LogInfo("{}", "Example shape print {1, 2, 3}.");
}

// Verifies that including "open3d/utility/Logging.h" does not pollute the
// global preprocessor namespace with LogError/LogWarning/LogInfo/LogDebug
// macros, allowing unrelated types and libraries to define or call methods
// with those names in the same translation unit.
TEST(Logging, NoGlobalMacroPollution) {
    UnrelatedLogger logger;
    EXPECT_EQ(logger.LogError(), 1);
    EXPECT_EQ(logger.LogWarning(), 2);
    EXPECT_EQ(logger.LogInfo(), 3);
    EXPECT_EQ(logger.LogDebug(), 4);
}

}  // namespace tests
}  // namespace open3d
