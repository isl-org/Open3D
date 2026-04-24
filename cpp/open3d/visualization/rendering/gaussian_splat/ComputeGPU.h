// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Generic GPU compute abstraction used by the Gaussian splatting pipeline.
// One header covers all platforms (OpenGL on Linux/Windows, Metal on macOS).
// Runtime shader resources are loaded from resources/gaussian_splat/.
//
// Typical usage:
//
//   // One GpuComputeFrame per geometry or composite stage (RAII).
//   GpuComputeFrame frame(ctx, GpuComputeFrame::kGeometry);
//
//   // Each dispatch: one temporary GpuComputePass expression.
//   GpuComputePass(ctx, ComputeProgramId::kGsProject, "gs_project")
//       .UBO(0, view_params_buf)
//       .SSBO(1, positions_buf)
//       .Dispatch(groups_x, 1, 1);
//   ctx.FullBarrier();

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace open3d {
namespace visualization {
namespace rendering {

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Program IDs for the Gaussian splatting compute shaders.
/// Values index into kGsShaderNames; kCount must stay last.
enum class ComputeProgramId : int {
    kGsProject = 0,
    kGsComposite = 1,
    kGsRadixHistograms = 2,
    kGsRadixScatter = 3,
    kGsDispatchArgs = 4,
    /// Merges GS linear depth (R32F) with Filament scene depth (reversed-Z)
    /// into a normalised R16UI texture for CPU readback.
    kGsDepthMerge = 5,
    kCount = 6,
};

/// Format selector for GaussianSplatGpuContext::BindImage().
enum class ImageFormat { kRGBA16F, kR32F, kR16UI };

/// Canonical shader base names indexed by ComputeProgramId.
/// Backends derive suffixes: GL SPIR-V → base+".spv", GL GLSL → base+".comp",
/// Metal entry point → base+"_main".
constexpr const char* kGsShaderNames[] = {
        "gaussian_project",
        "gaussian_composite",
        "gaussian_radix_sort_histograms",
        "gaussian_radix_sort",
        "gaussian_compute_dispatch_args",
        "gaussian_depth_merge",
};
static_assert(std::size(kGsShaderNames) ==
                      static_cast<std::size_t>(ComputeProgramId::kCount),
              "kGsShaderNames must match ComputeProgramId::kCount");

// ---------------------------------------------------------------------------
// GPU data layout structs
// ---------------------------------------------------------------------------

/// Per-radix-pass UBO at binding 14 (std140 layout, matches GPU shader).
/// Matches the RadixSortParams uniform block in the radix-sort shaders.
/// Named to match the GLSL block: `layout(std140, binding=14) uniform
/// RadixSortParams`.
struct RadixSortParams {
    std::uint32_t g_num_elements = 0;
    std::uint32_t g_shift = 0;
    std::uint32_t g_num_workgroups = 0;
    std::uint32_t g_num_blocks_per_workgroup = 0;
};
static_assert(sizeof(RadixSortParams) == 16,
              "RadixSortParams must be 16 bytes to match GLSL layout");

/// Per-view GPU resource handles (opaque: GL name or MTLBuffer/MTLTexture).
struct GaussianSplatViewGpuResources {
    std::uintptr_t view_params_buf = 0;
    std::uintptr_t positions_buf = 0;
    std::uintptr_t scales_buf = 0;
    std::uintptr_t rotations_buf = 0;
    std::uintptr_t dc_opacity_buf = 0;
    std::uintptr_t sh_buf = 0;
    /// Composite-pass projected data (32 B/splat, binding 6).
    /// Written by project, read only by composite.
    std::uintptr_t projected_composite_buf = 0;
    /// Per-frame work-stealing atomic counter (1 uint32): cleared before each
    /// composite pass; each composite workgroup atomically claims tile indices.
    std::uintptr_t tile_counts_buf = 0;
    /// GPU error/diagnostic counters (total_entries, error_flags, ...).
    std::uintptr_t counters_buf = 0;
    std::uintptr_t dispatch_args_buf = 0;
    /// Ping-pong sort buffers: keys = (tile<<D)|(depth>>T), values =
    /// splat_index.
    std::uintptr_t sort_keys_buf[2] = {0, 0};
    std::uintptr_t sort_values_buf[2] = {0, 0};
    std::uintptr_t histogram_buf = 0;
    std::uintptr_t radix_params_buf = 0;
    /// Bit-packed per-splat visibility mask. Bound at binding 15.
    std::uintptr_t mask_buf = 0;
    /// GS composite depth output (image binding 1); not the shared scene depth.
    std::uintptr_t composite_depth_tex = 0;
    /// Merged GS+Filament depth as R16UI (normalised [0,65535]); non-zero only
    /// when an offscreen capture is active and scene depth is available.
    std::uintptr_t merged_depth_u16_tex = 0;
    /// Which sort_values_buf index holds the final sorted splat indices.
    /// Set at the end of RunGaussianGeometryPasses; read by composite.
    int final_sort_src = 0;
    std::uint64_t cached_scene_id = 0;
    std::uint32_t cached_splat_count = 0;
    std::uint32_t warned_gpu_error_flags = 0;
};

// ---------------------------------------------------------------------------
// GaussianSplatGpuContext — abstract GPU backend interface
// ---------------------------------------------------------------------------

/// Platform-agnostic GPU operations for compute passes.
/// Implementations: CreateComputeGpuContextGL() (OpenGL) and
/// CreateComputeGpuContextMetal() (Metal).
class GaussianSplatGpuContext {
public:
    virtual ~GaussianSplatGpuContext() = default;

    /// Load all compute programs (lazy, idempotent).
    virtual bool EnsureProgramsLoaded() = 0;

    // --- Buffer management ------------------------------------------------
    virtual std::uintptr_t CreateBuffer(std::size_t size,
                                        const char* label = nullptr) = 0;
    virtual void DestroyBuffer(std::uintptr_t buf) = 0;
    /// Returns a valid handle (may replace buf when the API reallocates).
    virtual std::uintptr_t ResizeBuffer(std::uintptr_t buf,
                                        std::size_t new_size,
                                        const char* label = nullptr) = 0;

    /// GPU-private buffer: never CPU-mapped after creation.
    /// Metal: MTLStorageModePrivate; OpenGL: GL_DYNAMIC_COPY.
    virtual std::uintptr_t CreatePrivateBuffer(std::size_t size,
                                               const char* label = nullptr) {
        return CreateBuffer(size, label);
    }
    virtual std::uintptr_t ResizePrivateBuffer(std::uintptr_t buf,
                                               std::size_t new_size,
                                               const char* label = nullptr) {
        return ResizeBuffer(buf, new_size, label);
    }
    virtual void UploadBuffer(std::uintptr_t buf,
                              const void* data,
                              std::size_t size,
                              std::size_t offset) = 0;
    virtual bool DownloadBuffer(std::uintptr_t buf,
                                void* dst,
                                std::size_t size,
                                std::size_t offset) {
        (void)buf;
        (void)dst;
        (void)size;
        (void)offset;
        return false;
    }
    virtual void ClearBufferUInt32Zero(std::uintptr_t buf) = 0;

    // --- Bindings ---------------------------------------------------------
    virtual void BindSSBO(std::uint32_t binding, std::uintptr_t buf) = 0;
    virtual void BindUBO(std::uint32_t binding, std::uintptr_t buf) = 0;
    virtual void BindUBORange(std::uint32_t binding,
                              std::uintptr_t buf,
                              std::size_t offset,
                              std::size_t range_size) = 0;

    // --- Dispatch ---------------------------------------------------------
    virtual void UseProgram(ComputeProgramId id) = 0;
    virtual void Dispatch(std::uint32_t groups_x,
                          std::uint32_t groups_y,
                          std::uint32_t groups_z) = 0;
    virtual void DispatchIndirect(std::uintptr_t indirect_buf,
                                  std::size_t byte_offset) = 0;
    virtual void FullBarrier() = 0;

    /// Maximum compute dispatch group count along the X axis.
    /// Used to split large dispatches into 2D grids that stay within device
    /// limits (Vulkan: VkPhysicalDeviceLimits::maxComputeWorkGroupCount[0]).
    /// Default returns 65535 (Vulkan minimum guaranteed by the spec).
    virtual std::uint32_t GetMaxComputeWorkGroupCount() const { return 65535u; }

    // --- Textures / images ------------------------------------------------
    virtual std::uintptr_t CreateTexture2DR32F(std::uint32_t width,
                                               std::uint32_t height,
                                               const char* label = nullptr) = 0;
    virtual void DestroyTexture(std::uintptr_t tex) = 0;
    virtual std::uintptr_t ResizeTexture2DR32F(std::uintptr_t tex,
                                               std::uint32_t width,
                                               std::uint32_t height,
                                               const char* label = nullptr) = 0;
    /// Create or resize an R16UI texture for merged-depth CPU readback.
    virtual std::uintptr_t ResizeTexture2DR16UI(
            std::uintptr_t tex,
            std::uint32_t width,
            std::uint32_t height,
            const char* label = nullptr) = 0;

    /// Download the contents of an R32F texture into a float vector.
    /// The caller must ensure no outstanding GPU writes to the texture.
    /// Returns false when not supported or on error.
    virtual bool DownloadTextureR32F(std::uintptr_t tex,
                                     std::uint32_t width,
                                     std::uint32_t height,
                                     std::vector<float>& out) {
        (void)tex;
        (void)width;
        (void)height;
        (void)out;
        return false;
    }

    /// Download the contents of an R16UI texture into a uint16_t vector.
    /// The caller must ensure no outstanding GPU writes to the texture.
    /// Returns false when not supported or on error.
    virtual bool DownloadTextureR16UI(std::uintptr_t tex,
                                      std::uint32_t width,
                                      std::uint32_t height,
                                      std::vector<std::uint16_t>& out) {
        (void)tex;
        (void)width;
        (void)height;
        (void)out;
        return false;
    }

    /// Bind a write image at the given unit with the specified format.
    virtual void BindImage(std::uint32_t binding,
                           std::uintptr_t tex,
                           std::uint32_t width,
                           std::uint32_t height,
                           ImageFormat fmt) = 0;

    virtual void BindSamplerTexture(std::uint32_t unit,
                                    std::uintptr_t tex,
                                    std::uint32_t width,
                                    std::uint32_t height) = 0;

    // --- Frame sync -------------------------------------------------------
    /// Drain any pending GPU work (GL: glFinish; Metal: no-op, EndCompositePass
    /// already waits synchronously).
    virtual void FinishGpuWork() = 0;

    /// Returns whether the most recently submitted GPU work succeeded.
    virtual bool WasLastSubmitSuccessful() const { return true; }

    /// Metal: wraps each stage in a MTLCommandBuffer + encoder pair.
    /// OpenGL: no-ops (compute is issued inline on the current context).
    virtual void BeginGeometryPass() {}
    virtual void EndGeometryPass() {}
    virtual void BeginCompositePass() {}
    virtual void EndCompositePass() {}

    /// KHR_debug / Metal debug-group markers around each dispatch.
    /// Default implementations are no-ops.
    virtual void PushDebugGroup(const char* /*label*/) {}
    virtual void PopDebugGroup() {}
};

// ---------------------------------------------------------------------------
// GpuComputeFrame — RAII for Begin/EndGeometryPass or Begin/EndCompositePass
// ---------------------------------------------------------------------------

/// Wraps BeginGeometryPass()/EndGeometryPass() or the composite equivalents.
/// Ensures End() is always called even on early-return from a runner function.
class GpuComputeFrame {
public:
    enum Kind { kGeometry, kComposite };

    GpuComputeFrame(GaussianSplatGpuContext& ctx, Kind kind)
        : ctx_(ctx), kind_(kind) {
        if (kind_ == kGeometry) {
            ctx_.BeginGeometryPass();
        } else {
            ctx_.BeginCompositePass();
        }
    }
    ~GpuComputeFrame() { End(); }

    GpuComputeFrame(const GpuComputeFrame&) = delete;
    GpuComputeFrame& operator=(const GpuComputeFrame&) = delete;

    /// Explicitly end the frame early (dtor becomes a no-op).
    void End() {
        if (!ended_) {
            ended_ = true;
            if (kind_ == kGeometry) {
                ctx_.EndGeometryPass();
            } else {
                ctx_.EndCompositePass();
            }
        }
    }

private:
    GaussianSplatGpuContext& ctx_;
    Kind kind_;
    bool ended_ = false;
};

// ---------------------------------------------------------------------------
// GpuComputePass — RAII + builder for a single compute dispatch
// ---------------------------------------------------------------------------

/// RAII dispatch helper. Ctor: UseProgram + PushDebugGroup.
/// Builder methods bind resources (return *this for chaining).
/// Dispatch() / DispatchIndirect() fires the GPU work.
/// Dtor: PopDebugGroup (if programs loaded successfully).
///
/// If EnsureProgramsLoaded() fails, ok() returns false. All builder methods
/// and dispatch calls become no-ops — caller does NOT need to test ok() unless
/// it wants to short-circuit further work for the whole pass sequence.
class GpuComputePass {
public:
    GpuComputePass(GaussianSplatGpuContext& ctx,
                   ComputeProgramId pid,
                   const char* label = nullptr)
        : ctx_(ctx), label_(label) {
        ok_ = ctx_.EnsureProgramsLoaded();
        if (ok_) {
            ctx_.UseProgram(pid);
            if (label_) ctx_.PushDebugGroup(label_);
        }
    }

    ~GpuComputePass() {
        if (ok_ && label_) ctx_.PopDebugGroup();
    }

    GpuComputePass(const GpuComputePass&) = delete;
    GpuComputePass& operator=(const GpuComputePass&) = delete;

    /// Returns false only when EnsureProgramsLoaded() failed (device error).
    [[nodiscard]] bool ok() const { return ok_; }

    // --- Resource binding (fluent builder) --------------------------------

    GpuComputePass& UBO(std::uint32_t binding, std::uintptr_t buf) {
        if (ok_) ctx_.BindUBO(binding, buf);
        return *this;
    }

    GpuComputePass& UBORange(std::uint32_t binding,
                             std::uintptr_t buf,
                             std::size_t offset,
                             std::size_t size) {
        if (ok_) ctx_.BindUBORange(binding, buf, offset, size);
        return *this;
    }

    GpuComputePass& SSBO(std::uint32_t binding, std::uintptr_t buf) {
        if (ok_) ctx_.BindSSBO(binding, buf);
        return *this;
    }

    GpuComputePass& Image(std::uint32_t binding,
                          std::uintptr_t tex,
                          std::uint32_t w,
                          std::uint32_t h,
                          ImageFormat fmt) {
        if (ok_) ctx_.BindImage(binding, tex, w, h, fmt);
        return *this;
    }

    GpuComputePass& Sampler(std::uint32_t unit,
                            std::uintptr_t tex,
                            std::uint32_t w,
                            std::uint32_t h) {
        if (ok_) ctx_.BindSamplerTexture(unit, tex, w, h);
        return *this;
    }

    // --- Dispatch ---------------------------------------------------------

    void Dispatch(std::uint32_t gx, std::uint32_t gy, std::uint32_t gz) {
        if (ok_) ctx_.Dispatch(gx, gy, gz);
    }

    void DispatchIndirect(std::uintptr_t buf, std::size_t byte_offset) {
        if (ok_) ctx_.DispatchIndirect(buf, byte_offset);
    }

private:
    GaussianSplatGpuContext& ctx_;
    const char* label_;
    bool ok_;
};

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

#if !defined(__APPLE__)
// Vulkan-only: no GL compute factory.
#endif
#if defined(__APPLE__)
[[nodiscard]] std::unique_ptr<GaussianSplatGpuContext>
CreateComputeGpuContextMetal(std::uintptr_t device_handle,
                             std::uintptr_t command_queue_handle);
#endif

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
