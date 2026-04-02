// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Thin GPU abstraction for Gaussian splatting compute: one implementation uses
// OpenGL 4.6 compute + SPIR-V, the other uses Metal + prebuilt .metallib from
// the same GLSL source (via SPIRV-Cross). Buffer sizing and pass order live in
// GaussianComputePassRunner.cpp so backends do not fork dispatch rules.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace open3d {
namespace visualization {
namespace rendering {

/// Indices match kShaderSPVFiles / kMetalEntryNames table order.
enum class GaussianComputeProgramId : int {
    kProject = 0,
    kPrefixSum = 1,
    kScatter = 2,
    kComposite = 3,
    kRadixKeygen = 4,
    kRadixHistograms = 5,
    kRadixScatter = 6,
    kRadixPayload = 7,
    kComputeDispatchArgs = 8,
    kCount = 9,
};

/// RadixSortParams matches the std140 UBO at binding 14 in radix shaders.
struct RadixSortParamsGpu {
    std::uint32_t g_num_elements = 0;
    std::uint32_t g_shift = 0;
    std::uint32_t g_num_workgroups = 0;
    std::uint32_t g_num_blocks_per_workgroup = 0;
};

/// Per-view GPU resources (opaque handles: GL name or MTLBuffer / MTLTexture).
struct GaussianComputeViewGpuResources {
    std::uintptr_t view_params_buf = 0;
    std::uintptr_t positions_buf = 0;
    std::uintptr_t scales_buf = 0;
    std::uintptr_t rotations_buf = 0;
    std::uintptr_t dc_opacity_buf = 0;
    std::uintptr_t sh_buf = 0;
    std::uintptr_t projected_buf = 0;
    std::uintptr_t tile_counts_buf = 0;
    std::uintptr_t tile_offsets_buf = 0;
    std::uintptr_t tile_heads_buf = 0;
    std::uintptr_t counters_buf = 0;
    std::uintptr_t tile_entries_buf = 0;
    std::uintptr_t dispatch_args_buf = 0;
    std::uintptr_t sort_keys_buf[2] = {0, 0};
    std::uintptr_t sort_values_buf[2] = {0, 0};
    std::uintptr_t histogram_buf = 0;
    std::uintptr_t radix_params_buf = 0;
    std::uintptr_t sorted_entries_buf = 0;
    /// GS composite depth output (image binding 1); not the shared scene depth.
    std::uintptr_t composite_depth_tex = 0;
    std::uint64_t cached_scene_id = 0;
    std::uint32_t cached_splat_count = 0;
};

/// Backend-specific GPU operations used by GaussianComputePassRunner.
class GaussianComputeGpuContext {
public:
    virtual ~GaussianComputeGpuContext() = default;

    virtual bool EnsureProgramsLoaded() = 0;

    virtual std::uintptr_t CreateBuffer(std::size_t size) = 0;
    virtual void DestroyBuffer(std::uintptr_t buf) = 0;
    /// Returns a valid handle (may replace buf when the API reallocates).
    virtual std::uintptr_t ResizeBuffer(std::uintptr_t buf,
                                        std::size_t new_size) = 0;

    /// Create / resize a GPU-private buffer: one that is never CPU-mapped after
    /// creation and is only accessed by GPU compute passes.
    /// Metal: MTLStorageModePrivate; OpenGL: GL_DYNAMIC_COPY.
    /// Default: falls back to CreateBuffer / ResizeBuffer.
    virtual std::uintptr_t CreatePrivateBuffer(std::size_t size) {
        return CreateBuffer(size);
    }
    virtual std::uintptr_t ResizePrivateBuffer(std::uintptr_t buf,
                                               std::size_t new_size) {
        return ResizeBuffer(buf, new_size);
    }
    virtual void UploadBuffer(std::uintptr_t buf,
                              const void* data,
                              std::size_t size,
                              std::size_t offset) = 0;
    virtual void ClearBufferUInt32Zero(std::uintptr_t buf) = 0;

    virtual void BindSSBO(std::uint32_t binding, std::uintptr_t buf) = 0;
    virtual void BindUBO(std::uint32_t binding, std::uintptr_t buf) = 0;
    virtual void BindUBORange(std::uint32_t binding,
                              std::uintptr_t buf,
                              std::size_t offset,
                              std::size_t range_size) = 0;

    virtual void UseProgram(GaussianComputeProgramId id) = 0;
    virtual void Dispatch(std::uint32_t groups_x,
                          std::uint32_t groups_y,
                          std::uint32_t groups_z) = 0;
    virtual void DispatchIndirect(std::uintptr_t indirect_buf,
                                  std::size_t byte_offset) = 0;
    virtual void FullBarrier() = 0;
    virtual std::uintptr_t CreateTexture2DR32F(std::uint32_t width,
                                               std::uint32_t height) = 0;
    virtual void DestroyTexture(std::uintptr_t tex) = 0;
    virtual std::uintptr_t ResizeTexture2DR32F(std::uintptr_t tex,
                                               std::uint32_t width,
                                               std::uint32_t height) = 0;

    virtual void BindImageRGBA16FWrite(std::uint32_t binding,
                                       std::uintptr_t tex,
                                       std::uint32_t width,
                                       std::uint32_t height) = 0;
    virtual void BindImageR32FWrite(std::uint32_t binding,
                                    std::uintptr_t tex,
                                    std::uint32_t width,
                                    std::uint32_t height) = 0;
    virtual void BindSamplerTexture(std::uint32_t unit,
                                    std::uintptr_t tex,
                                    std::uint32_t width,
                                    std::uint32_t height) = 0;

    /// Called at the end of the composite pass to drain any pending GPU work.
    /// On OpenGL, blocks via glFinish(). On Metal, EndCompositePass() already
    /// waits synchronously, so this is a no-op.
    virtual void FinishGpuWork() = 0;

    /// Returns whether the most recently submitted GPU work completed without
    /// an execution error.
    virtual bool WasLastSubmitSuccessful() const { return true; }

    /// Metal: wraps each compute pass in a MTLCommandBuffer + encoder pair.
    /// OpenGL implementations are no-ops (compute is issued inline).
    virtual void BeginGeometryPass() {}
    virtual void EndGeometryPass() {}

    virtual void BeginCompositePass() {}
    virtual void EndCompositePass() {}
};

#if !defined(__APPLE__)
/// Builds the OpenGL + SPIR-V GaussianComputeGpuContext implementation.
std::unique_ptr<GaussianComputeGpuContext>
CreateGaussianComputeGpuContextOpenGL();
#endif
#if defined(__APPLE__)
std::unique_ptr<GaussianComputeGpuContext> CreateGaussianComputeGpuContextMetal(
        std::uintptr_t device_handle, std::uintptr_t command_queue_handle);
#endif

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
