// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstring>
#include <string>

#ifdef BUILD_CUDA_MODULE
#include "open3d/core/CUDAUtils.h"
#endif

#include "open3d/Open3D.h"
#include "tests/Tests.h"

#ifdef BUILD_CUDA_MODULE
/// Returns true if --disable_p2p flag is used.
bool ShallDisableP2P(int argc, char** argv) {
    bool shall_disable_p2p = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--disable_p2p") == 0) {
            shall_disable_p2p = true;
            break;
        }
    }
    return shall_disable_p2p;
}
#endif

int main(int argc, char** argv) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    utility::CompilerInfo::GetInstance().Print();
    utility::CPUInfo::GetInstance().Print();
    utility::ISAInfo::GetInstance().Print();

#ifdef BUILD_CUDA_MODULE
    if (ShallDisableP2P(argc, argv)) {
        core::CUDAState::GetInstance().ForceDisableP2PForTesting();
        utility::LogInfo("P2P device transfer has been disabled.");
    }
#endif

    testing::InitGoogleMock(&argc, argv);
    return RUN_ALL_TESTS();
}
