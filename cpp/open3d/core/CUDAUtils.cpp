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

#include "open3d/core/CUDAUtils.h"

#include "open3d/utility/Console.h"

#ifdef BUILD_CUDA_MODULE
#include "open3d/core/CUDAState.cuh"
#include "open3d/core/MemoryManager.h"
#endif

namespace open3d {
namespace core {
namespace cuda {

int DeviceCount() {
#ifdef BUILD_CUDA_MODULE
    try {
        std::shared_ptr<CUDAState> cuda_state = CUDAState::GetInstance();
        return cuda_state->GetNumDevices();
    } catch (const std::runtime_error& e) {  // GetInstance can throw
        return 0;
    }
#else
    return 0;
#endif
}

bool IsAvailable() { return cuda::DeviceCount() > 0; }

void ReleaseCache() {
#ifdef BUILD_CUDA_MODULE
#ifdef BUILD_CACHED_CUDA_MANAGER
    CUDACachedMemoryManager::ReleaseCache();
#else
    utility::LogWarning(
            "Built without cached CUDA memory manager, cuda::ReleaseCache() "
            "has no effect.");
#endif

#else
    utility::LogWarning("Built without CUDA module, cuda::ReleaseCache().");
#endif
}

}  // namespace cuda
}  // namespace core
}  // namespace open3d

// C interface to provide un-mangled function to Python ctypes
extern "C" int open3d_core_cuda_device_count() {
    return open3d::core::cuda::DeviceCount();
}
