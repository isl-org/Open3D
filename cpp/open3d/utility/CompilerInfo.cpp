// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/CompilerInfo.h"

#include <memory>
#include <string>

#include "open3d/utility/Logging.h"

namespace open3d {
namespace utility {

CompilerInfo::CompilerInfo() {}

CompilerInfo& CompilerInfo::GetInstance() {
    static CompilerInfo instance;
    return instance;
}

std::string CompilerInfo::CXXStandard() const {
    return std::string(OPEN3D_CXX_STANDARD);
}

std::string CompilerInfo::CXXCompilerId() const {
    return std::string(OPEN3D_CXX_COMPILER_ID);
}

std::string CompilerInfo::CXXCompilerVersion() const {
    return std::string(OPEN3D_CXX_COMPILER_VERSION);
}

std::string CompilerInfo::CUDACompilerId() const {
    return std::string(OPEN3D_CUDA_COMPILER_ID);
}

std::string CompilerInfo::CUDACompilerVersion() const {
    return std::string(OPEN3D_CUDA_COMPILER_VERSION);
}

void CompilerInfo::Print() const {
#ifdef BUILD_CUDA_MODULE
    utility::LogInfo("CompilerInfo: C++ {}, {} {}, {} {}.", CXXStandard(),
                     CXXCompilerId(), CXXCompilerVersion(), CUDACompilerId(),
                     CUDACompilerVersion());
#else
    utility::LogInfo("CompilerInfo: C++ {}, {} {}, CUDA disabled.",
                     CXXStandard(), CXXCompilerId(), CXXCompilerVersion());
#endif
}

}  // namespace utility
}  // namespace open3d
