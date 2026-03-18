// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace open3d {
namespace visualization {
namespace rendering {

struct MetalComputePipelineHandle {
    std::uintptr_t library = 0;
    std::uintptr_t function = 0;
    std::uintptr_t pipeline = 0;
    bool valid = false;
};

struct MetalComputeDispatch {
    MetalComputePipelineHandle pipeline;
    std::uint32_t group_count_x = 1;
    std::uint32_t group_count_y = 1;
    std::uint32_t group_count_z = 1;
    std::uint32_t thread_count_x = 1;
    std::uint32_t thread_count_y = 1;
    std::uint32_t thread_count_z = 1;
};

MetalComputePipelineHandle CompileMetalComputePipeline(
        std::uintptr_t device_handle,
        const std::string& source,
        const std::string& entry_point,
        const std::string& label,
        std::string* error_message);

void DestroyMetalComputePipeline(MetalComputePipelineHandle handle);

bool DispatchMetalComputePipelines(
        std::uintptr_t command_queue_handle,
        const std::vector<MetalComputeDispatch>& dispatches,
        std::string* error_message);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d