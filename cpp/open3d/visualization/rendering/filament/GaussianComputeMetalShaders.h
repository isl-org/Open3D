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

struct MetalBufferHandle {
    std::uintptr_t buffer = 0;
    std::size_t size = 0;
    void* mapped = nullptr;
    bool valid = false;
};

struct MetalBufferBinding {
    std::uint32_t index = 0;
    MetalBufferHandle buffer;
    std::size_t offset = 0;
};

struct MetalComputeDispatch {
    MetalComputePipelineHandle pipeline;
    std::vector<MetalBufferBinding> buffers;
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

// Metal shared buffer management (host-visible, coherent).
MetalBufferHandle CreateMetalSharedBuffer(std::uintptr_t device_handle,
                                          std::size_t size,
                                          const std::string& label,
                                          std::string* error_message);

void DestroyMetalSharedBuffer(MetalBufferHandle handle);

bool UploadMetalSharedBuffer(const MetalBufferHandle& handle,
                             const void* data,
                             std::size_t size,
                             std::size_t offset,
                             std::string* error_message);

bool DownloadMetalSharedBuffer(const MetalBufferHandle& handle,
                               void* data,
                               std::size_t size,
                               std::size_t offset,
                               std::string* error_message);

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d