// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>

#include "open3d/Open3D.h"

int main(int argc, char **argv) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    utility::LogDebug("This Debug message should be visible, {} {:.2f}",
                      "format:", 0.42001);
    utility::LogInfo("This Info message should be visible, {} {:.2f}",
                     "format:", 0.42001);
    utility::LogWarning("This Warning message should be visible, {} {:.2f}",
                        "format:", 0.42001);

    utility::SetVerbosityLevel(utility::VerbosityLevel::Info);

    utility::LogDebug("This Debug message should NOT be visible, {} {:.2f}",
                      "format:", 0.42001);
    utility::LogInfo("This Info message should be visible, {} {:.2f}",
                     "format:", 0.42001);
    utility::LogWarning("This Warning message should be visible, {} {:.2f}",
                        "format:", 0.42001);

    try {
        utility::LogError("This Error exception is caught");
    } catch (const std::exception &e) {
        utility::LogInfo("Caught exception msg: {}", e.what());
    }
    utility::LogInfo("This Info message shall print in regular color");
    utility::LogError("This Error message terminates the program");
    utility::LogError("This Error message should NOT be visible");

    return 0;
}
