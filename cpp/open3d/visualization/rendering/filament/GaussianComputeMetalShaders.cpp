// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianComputeMetalShaders.h"

#include <vector>

namespace open3d {
namespace visualization {
namespace rendering {

MetalComputePipelineHandle CompileMetalComputePipeline(
        std::uintptr_t,
        const std::string&,
        const std::string&,
        const std::string&,
        std::string* error_message) {
    if (error_message) {
        *error_message =
                "Metal compute pipelines are only available on Apple "
                "platforms.";
    }
    return {};
}

void DestroyMetalComputePipeline(MetalComputePipelineHandle) {}

bool DispatchMetalComputePipelines(std::uintptr_t,
                                   const std::vector<MetalComputeDispatch>&,
                                   std::string* error_message) {
    if (error_message) {
        *error_message =
                "Metal compute dispatch is only available on Apple platforms.";
    }
    return false;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d