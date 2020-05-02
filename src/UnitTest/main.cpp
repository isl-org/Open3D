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

#include <gtest/gtest.h>
#include <cstring>
#include <string>

#ifdef BUILD_CUDA_MODULE
#include "Open3D/Core/CUDAState.cuh"
#endif

#include "Open3D/Utility/Console.h"
#include "TestUtility/Print.h"
#include "TestUtility/Rand.h"
#include "TestUtility/Raw.h"

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
#ifdef BUILD_CUDA_MODULE
    if (ShallDisableP2P(argc, argv)) {
        std::shared_ptr<open3d::CUDAState> cuda_state =
                open3d::CUDAState::GetInstance();
        cuda_state->ForceDisableP2PForTesting();
        open3d::utility::LogInfo("P2P device transfer has been disabled.");
    }
#endif
    testing::InitGoogleTest(&argc, argv);
    open3d::utility::SetVerbosityLevel(open3d::utility::VerbosityLevel::Debug);
    return RUN_ALL_TESTS();
}
