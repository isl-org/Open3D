// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Metal implementation of GaussianSplatGpuContext.
// Compiled only on Apple platforms.

#include "open3d/visualization/rendering/gaussian_splat/ComputeGPU.h"

#if defined(__APPLE__)

#import <Metal/Metal.h>

#include <array>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"

namespace open3d {
namespace visualization {
namespace rendering {

namespace {

/// Threads per threadgroup matching GLSL local_size_* in each shader.
/// Must have exactly ComputeProgramId::kCount rows — enforced by static_assert.
static const NSUInteger kThreadsPerGroup[][3] = {
        {64, 1, 1},   // kGsProject
        {256, 1, 1},  // kGsPrefixSum
        {64, 1, 1},   // kGsScatter
        {16, 16, 1},  // kGsComposite
        {256, 1, 1},  // kGsRadixKeygen
        {256, 1, 1},  // kGsRadixHistograms
        {256, 1, 1},  // kGsRadixScatter
        {256, 1, 1},  // kGsRadixPayload
        {1, 1, 1},    // kGsDispatchArgs
        {16, 16, 1},  // kGsDepthMerge
        {256, 1, 1},  // kGsOneSweepGlobalHist
        {256, 1, 1},  // kGsOneSweepDigitPass
};
static_assert(std::size(kThreadsPerGroup) ==
                      static_cast<std::size_t>(ComputeProgramId::kCount),
              "kThreadsPerGroup row count must match ComputeProgramId::kCount");

static constexpr std::size_t kMaxBindings = 32;

}  // namespace

class GaussianSplatGpuContextMetal final : public GaussianSplatGpuContext {
public:
    GaussianSplatGpuContextMetal(std::uintptr_t device_handle,
                                 std::uintptr_t queue_handle)
        : device_((__bridge id<MTLDevice>)reinterpret_cast<void*>(
                  device_handle)),
          queue_((__bridge id<MTLCommandQueue>)reinterpret_cast<void*>(
                  queue_handle)) {}

    ~GaussianSplatGpuContextMetal() override {
        // Drain any in-flight command buffers. EndCompositePass() already waits
        // synchronously, so last_submitted_comp_cb_ is complete by the time we
        // reach the destructor. The wait below is a no-cost safety net.
        if (encoder_) {
            [encoder_ endEncoding];
            encoder_ = nil;
        }
        if (geom_cb_) {
            [geom_cb_ waitUntilCompleted];
            geom_cb_ = nil;
        }
        if (last_submitted_comp_cb_) {
            [last_submitted_comp_cb_ waitUntilCompleted];
            last_submitted_comp_cb_ = nil;
        }
        for (auto& p : pipelines_) {
            p = nil;
        }
        library_ = nil;
        sampler_ = nil;
    }

    bool EnsureProgramsLoaded() override {
        if (library_) {
            return true;
        }
        const std::string lib_path = EngineInstance::GetResourcePath() +
                                     "/gaussian_splat/gaussian_splat.metallib";
        if (!utility::filesystem::FileExists(lib_path)) {
            utility::LogWarning("Metal: missing metallib at {}", lib_path);
            return false;
        }
        NSString* npath = [NSString stringWithUTF8String:lib_path.c_str()];
        NSError* err = nil;
        library_ = [device_ newLibraryWithFile:npath error:&err];
        if (!library_) {
            utility::LogWarning(
                    "Metal: newLibraryWithFile failed: {}",
                    err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }

        auto load_pipeline = [&](int i) -> bool {
            std::string entry_name = std::string(kGsShaderNames[i]) + "_main";
            NSString* name = [NSString stringWithUTF8String:entry_name.c_str()];
            id<MTLFunction> fn = [library_ newFunctionWithName:name];
            if (!fn) {
                utility::LogDebug("Metal: missing function {}", entry_name);
                return false;
            }
            NSError* perr = nil;
            id<MTLComputePipelineState> ps =
                    [device_ newComputePipelineStateWithFunction:fn
                                                           error:&perr];
            if (!ps) {
                utility::LogDebug(
                        "Metal: pipeline {} failed: {}", entry_name,
                        perr ? [[perr localizedDescription] UTF8String] : "?");
                return false;
            }
            pipelines_[i] = ps;
            return true;
        };

        // Phase 1: base programs (required). Failure is fatal for Metal GS.
        for (int i = 0; i < kGsFirstOneSweepProgram; ++i) {
            if (!load_pipeline(i)) {
                utility::LogWarning("Metal: failed to load base GS program {}",
                                    kGsShaderNames[i]);
                return false;
            }
        }

        // Phase 2: OneSweep programs (optional). Failure disables OneSweep
        // but does not affect the base radix-sort pipeline.
        // SIMD-group arithmetic is always available on A11+ / M-series;
        // pipeline compilation below acts as the implicit capability guard.
        onesweep_programs_valid_ = true;
        for (int i = kGsFirstOneSweepProgram;
             i < static_cast<int>(ComputeProgramId::kCount); ++i) {
            if (!load_pipeline(i)) {
                onesweep_programs_valid_ = false;
                utility::LogDebug("Metal: OneSweep program {} unavailable; "
                                  "falling back to radix sort.",
                                  kGsShaderNames[i]);
                break;
            }
        }
        if (onesweep_programs_valid_) {
            utility::LogDebug("Metal: OneSweep programs loaded.");
        }

        @autoreleasepool {
            MTLSamplerDescriptor* sdesc = [[MTLSamplerDescriptor alloc] init];
            sdesc.minFilter = MTLSamplerMinMagFilterLinear;
            sdesc.magFilter = MTLSamplerMinMagFilterLinear;
            sdesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
            sdesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
            sampler_ = [device_ newSamplerStateWithDescriptor:sdesc];
        }
        return true;
    }

    bool AreOneSweepProgramsLoaded() const override {
        return onesweep_programs_valid_;
    }

    std::uintptr_t CreateBuffer(std::size_t size,
                                const char* label = nullptr) override {
        return AllocateBuffer(size, MTLResourceStorageModeShared, label);
    }

    /// GPU-private buffer: not CPU-accessible, optimal for GPU-only
    /// intermediates.
    std::uintptr_t CreatePrivateBuffer(std::size_t size,
                                       const char* label = nullptr) override {
        return AllocateBuffer(size, MTLResourceStorageModePrivate, label);
    }

    void DestroyBuffer(std::uintptr_t buf) override {
        if (buf == 0) {
            return;
        }
        buffer_sizes_.erase(buf);
        id<MTLBuffer> b =
                (__bridge_transfer id<MTLBuffer>)reinterpret_cast<void*>(buf);
        (void)b;
    }

    std::uintptr_t ResizeBuffer(std::uintptr_t buf,
                                std::size_t new_size,
                                const char* label = nullptr) override {
        return ReallocBuffer(buf, new_size, /*priv=*/false, label);
    }

    std::uintptr_t ResizePrivateBuffer(std::uintptr_t buf,
                                       std::size_t new_size,
                                       const char* label = nullptr) override {
        return ReallocBuffer(buf, new_size, /*priv=*/true, label);
    }

    void UploadBuffer(std::uintptr_t buf,
                      const void* data,
                      std::size_t size,
                      std::size_t offset) override {
        if (!buf || !data || size == 0) {
            return;
        }
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(buf);
        std::memcpy(static_cast<char*>([b contents]) + offset, data, size);
    }

    bool DownloadBuffer(std::uintptr_t buf,
                        void* dst,
                        std::size_t size,
                        std::size_t offset) override {
        if (!buf || !dst || size == 0) {
            return false;
        }
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(buf);
        if ([b storageMode] == MTLStorageModePrivate) {
            return false;
        }
        std::memcpy(dst, static_cast<const char*>([b contents]) + offset, size);
        return true;
    }

    void ClearBufferUInt32Zero(std::uintptr_t buf) override {
        auto it = buffer_sizes_.find(buf);
        if (it == buffer_sizes_.end()) {
            return;
        }
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)reinterpret_cast<void*>(buf);
        const NSUInteger size = static_cast<NSUInteger>(it->second);

        // When a compute encoder is active we are inside a pass:
        // clears must happen on the GPU timeline between dispatches,
        // not at CPU encode time. Use a blit encoder so the fill
        // executes at the correct point in the command buffer.
        id<MTLCommandBuffer> cb = geom_cb_ ? geom_cb_ : comp_cb_;
        if (encoder_ && cb) {
            [encoder_ endEncoding];
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit fillBuffer:b range:NSMakeRange(0, size) value:0];
            [blit endEncoding];
            encoder_ = [cb computeCommandEncoder];
            ApplyEncoderState();
            return;
        }

        // No active pass — CPU memset is safe (before CB commit).
        std::memset([b contents], 0, it -> second);
    }

    void BindSSBO(std::uint32_t binding, std::uintptr_t buf) override {
        id<MTLBuffer> b =
                buf ? (__bridge id<MTLBuffer>)reinterpret_cast<void*>(buf)
                    : nil;
        SetBufferBinding(binding, b, 0);
    }

    void BindUBO(std::uint32_t binding, std::uintptr_t buf) override {
        BindSSBO(binding, buf);
    }

    void BindUBORange(std::uint32_t binding,
                      std::uintptr_t buf,
                      std::size_t offset,
                      std::size_t range_size) override {
        (void)range_size;
        id<MTLBuffer> b =
                buf ? (__bridge id<MTLBuffer>)reinterpret_cast<void*>(buf)
                    : nil;
        SetBufferBinding(binding, b, static_cast<NSUInteger>(offset));
    }

    void UseProgram(ComputeProgramId id) override {
        current_program_ = static_cast<int>(id);
        if (!encoder_) {
            return;
        }
        int i = static_cast<int>(id);
        if (i >= 0 && i < static_cast<int>(ComputeProgramId::kCount)) {
            [encoder_ setComputePipelineState:pipelines_[i]];
        }
    }

    void PushDebugGroup(const char* label) override {
        if (encoder_) {
            [encoder_ pushDebugGroup:[NSString stringWithUTF8String:label]];
        }
    }
    void PopDebugGroup() override {
        if (encoder_) {
            [encoder_ popDebugGroup];
        }
    }

    void Dispatch(std::uint32_t groups_x,
                  std::uint32_t groups_y,
                  std::uint32_t groups_z) override {
        if (!encoder_) {
            return;
        }
        int pi = current_program_;
        if (pi < 0 || pi >= static_cast<int>(ComputeProgramId::kCount)) {
            return;
        }
        MTLSize tg =
                MTLSizeMake(kThreadsPerGroup[pi][0], kThreadsPerGroup[pi][1],
                            kThreadsPerGroup[pi][2]);
        // dispatchThreads: (Metal 2 / macOS 10.13+) lets the driver handle the
        // partial last threadgroup automatically; total = workgroup_count *
        // tpg.
        MTLSize total = MTLSizeMake(
                static_cast<NSUInteger>(std::max(1u, groups_x)) * tg.width,
                static_cast<NSUInteger>(std::max(1u, groups_y)) * tg.height,
                static_cast<NSUInteger>(std::max(1u, groups_z)) * tg.depth);
        [encoder_ dispatchThreads:total threadsPerThreadgroup:tg];
    }

    void DispatchIndirect(std::uintptr_t indirect_buf,
                          std::size_t byte_offset) override {
        if (!encoder_ || indirect_buf == 0) {
            return;
        }
        int pi = current_program_;
        if (pi < 0 || pi >= static_cast<int>(ComputeProgramId::kCount)) {
            return;
        }
        id<MTLBuffer> ib =
                (__bridge id<MTLBuffer>)reinterpret_cast<void*>(indirect_buf);
        MTLSize tg =
                MTLSizeMake(kThreadsPerGroup[pi][0], kThreadsPerGroup[pi][1],
                            kThreadsPerGroup[pi][2]);
        [encoder_ dispatchThreadgroupsWithIndirectBuffer:ib
                                    indirectBufferOffset:byte_offset
                                   threadsPerThreadgroup:tg];
    }

    void FullBarrier() override {
        if (!encoder_) {
            return;
        }
        // memoryBarrierWithScope: available since Metal 2 (macOS 10.14+).
        // macOS 10.14 is the minimum supported since Xcode 12 dropped 10.13.
        if (@available(macOS 10.14, *)) {
            [encoder_ memoryBarrierWithScope:MTLBarrierScopeBuffers |
                                             MTLBarrierScopeTextures];
            return;
        }
        // Fallback on very old hardware: commit and restart the encoder so all
        // prior writes are visible to subsequent dispatches.
        id<MTLCommandBuffer> cb = geom_cb_ ? geom_cb_ : comp_cb_;
        if (cb) {
            RestartComputeEncoder(cb, /*restore_state=*/true);
        }
    }

    std::uintptr_t CreateTexture2DR32F(std::uint32_t width,
                                       std::uint32_t height,
                                       const char* label = nullptr) override {
        MTLTextureDescriptor* d = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                             width:width
                                            height:height
                                         mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        d.storageMode = MTLStorageModePrivate;
        id<MTLTexture> t = [device_ newTextureWithDescriptor:d];
        if (!t) {
            return 0;
        }
        if (label && label[0] != '\0') {
            t.label = [NSString stringWithUTF8String:label];
        }
        texture_sizes_[reinterpret_cast<std::uintptr_t>(t)] = {width, height};
        return reinterpret_cast<std::uintptr_t>((__bridge_retained void*)t);
    }

    void DestroyTexture(std::uintptr_t tex) override {
        if (tex == 0) {
            return;
        }
        texture_sizes_.erase(tex);
        id<MTLTexture> t =
                (__bridge_transfer id<MTLTexture>)reinterpret_cast<void*>(tex);
        (void)t;
    }

    std::uintptr_t ResizeTexture2DR32F(std::uintptr_t tex,
                                       std::uint32_t width,
                                       std::uint32_t height,
                                       const char* label = nullptr) override {
        if (tex == 0) {
            return CreateTexture2DR32F(width, height, label);
        }
        DestroyTexture(tex);
        return CreateTexture2DR32F(width, height, label);
    }

    std::uintptr_t ResizeTexture2DR16UI(std::uintptr_t tex,
                                        std::uint32_t width,
                                        std::uint32_t height,
                                        const char* label = nullptr) override {
        if (tex != 0) {
            DestroyTexture(tex);
        }
        // Allocate a CPU-visible R16Uint texture for merged depth readback.
        MTLTextureDescriptor* d = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Uint
                                             width:width
                                            height:height
                                         mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        d.storageMode = MTLStorageModePrivate;
        id<MTLTexture> t = [device_ newTextureWithDescriptor:d];
        if (!t) {
            return 0;
        }
        if (label && label[0] != '\0') {
            t.label = [NSString stringWithUTF8String:label];
        }
        const std::uintptr_t k =
                reinterpret_cast<std::uintptr_t>((__bridge_retained void*)t);
        texture_sizes_[k] = {width, height};
        return k;
    }

    bool DownloadTextureR32F(std::uintptr_t tex,
                             std::uint32_t width,
                             std::uint32_t height,
                             std::vector<float>& out) override {
        if (tex == 0 || width == 0 || height == 0) return false;
        id<MTLTexture> src =
                (__bridge id<MTLTexture>)reinterpret_cast<void*>(tex);
        return DownloadTextureImpl(src, width, height, sizeof(float), &out,
                                   nullptr);
    }

    bool DownloadTextureR16UI(std::uintptr_t tex,
                              std::uint32_t width,
                              std::uint32_t height,
                              std::vector<std::uint16_t>& out) override {
        if (tex == 0 || width == 0 || height == 0) return false;
        id<MTLTexture> src =
                (__bridge id<MTLTexture>)reinterpret_cast<void*>(tex);
        return DownloadTextureImpl(src, width, height, sizeof(std::uint16_t),
                                   nullptr, &out);
    }

    void BindImage(std::uint32_t binding,
                   std::uintptr_t tex,
                   std::uint32_t /*width*/,
                   std::uint32_t /*height*/,
                   ImageFormat /*fmt*/) override {
        // Metal infers format and access from the shader; w/h/fmt are unused.
        id<MTLTexture> t =
                tex ? (__bridge id<MTLTexture>)reinterpret_cast<void*>(tex)
                    : nil;
        SetTextureBinding(binding, t);
    }

    void BindSamplerTexture(std::uint32_t unit,
                            std::uintptr_t tex,
                            std::uint32_t /*width*/,
                            std::uint32_t /*height*/) override {
        id<MTLTexture> t =
                tex ? (__bridge id<MTLTexture>)reinterpret_cast<void*>(tex)
                    : nil;
        SetTextureBinding(unit, t);
        SetSamplerBinding(unit, sampler_);
    }

    void FinishGpuWork() override {}  // EndCompositePass() already waits.

    bool WasLastSubmitSuccessful() const override {
        return last_submit_succeeded_;
    }

    void BeginGeometryPass() override {
        EndGeometryPass();
        last_submit_succeeded_ = true;
        ResetPassState();
        geom_cb_ = [queue_ commandBuffer];
        encoder_ = [geom_cb_ computeCommandEncoder];
    }

    void EndGeometryPass() override {
        if (encoder_) {
            [encoder_ endEncoding];
            encoder_ = nil;
        }
        if (geom_cb_) {
            // Commit without waiting: the geometry CB and the subsequent
            // Filament frame CB execute in submission order on the same queue,
            // so Filament's mesh rendering will not begin until geometry
            // is complete on the GPU.
            [geom_cb_ commit];
            geom_cb_ = nil;
        }
    }

    void BeginCompositePass() override {
        if (encoder_) {
            [encoder_ endEncoding];
            encoder_ = nil;
        }
        last_submit_succeeded_ = true;
        ResetPassState();
        comp_cb_ = [queue_ commandBuffer];
        encoder_ = [comp_cb_ computeCommandEncoder];
    }

    void EndCompositePass() override {
        if (encoder_) {
            [encoder_ endEncoding];
            encoder_ = nil;
        }
        if (comp_cb_) {
            id<MTLCommandBuffer> cb = comp_cb_;
            comp_cb_ = nil;
            [cb commit];
            // Block the CPU until the GPU composite finishes — matches the
            // OpenGL path which calls glFinish() before EndCompositePass().
            [cb waitUntilCompleted];
            const bool success = [cb status] != MTLCommandBufferStatusError;
            last_submit_succeeded_ = success;
            if (!success) {
                NSError* err = [cb error];
                utility::LogWarning(
                        "Metal composite CB error: {}",
                        err ? [[err localizedDescription] UTF8String]
                            : "unknown");
            }
            // Keep a strong ref so the destructor's drain is a no-cost wait.
            last_submitted_comp_cb_ = cb;
        }
    }

private:
    struct BufferBindingState {
        id<MTLBuffer> buffer = nil;
        NSUInteger offset = 0;
        bool is_bound = false;
    };
    struct TextureBindingState {
        id<MTLTexture> texture = nil;
        bool is_bound = false;
    };
    struct SamplerBindingState {
        id<MTLSamplerState> sampler = nil;
        bool is_bound = false;
    };

    // Download a private MTLTexture into a CPU vector using a blit encoder.
    // Exactly one of f32_out and u16_out must be non-null.
    bool DownloadTextureImpl(id<MTLTexture> src,
                             std::uint32_t width,
                             std::uint32_t height,
                             std::size_t bytes_per_pixel,
                             std::vector<float>* f32_out,
                             std::vector<std::uint16_t>* u16_out) {
        if (!src || !queue_) return false;
        const NSUInteger row_bytes =
                static_cast<NSUInteger>(width) * bytes_per_pixel;
        const NSUInteger total_bytes = row_bytes * height;
        // Allocate a shared (CPU-visible) staging buffer.
        id<MTLBuffer> staging =
                [device_ newBufferWithLength:total_bytes
                                     options:MTLResourceStorageModeShared];
        if (!staging) return false;

        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        if (!cb) return false;
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromTexture:src
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(width, height, 1)
                                toBuffer:staging
                       destinationOffset:0
                  destinationBytesPerRow:row_bytes
                destinationBytesPerImage:total_bytes];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        const void* ptr = [staging contents];
        if (!ptr) return false;
        if (f32_out) {
            f32_out->resize(static_cast<std::size_t>(width) * height);
            std::memcpy(f32_out->data(), ptr, total_bytes);
        } else if (u16_out) {
            u16_out->resize(static_cast<std::size_t>(width) * height);
            std::memcpy(u16_out->data(), ptr, total_bytes);
        }
        return true;
    }

    std::uintptr_t AllocateBuffer(std::size_t size,
                                  MTLResourceOptions opts,
                                  const char* label) {
        if (size == 0) {
            return 0;
        }
        id<MTLBuffer> buf = [device_ newBufferWithLength:size options:opts];
        if (!buf) {
            return 0;
        }
        if (label != nullptr && label[0] != '\0') {
            [buf setLabel:[NSString stringWithUTF8String:label]];
        }
        std::uintptr_t key =
                reinterpret_cast<std::uintptr_t>((__bridge_retained void*)buf);
        buffer_sizes_[key] = size;
        return key;
    }

    std::uintptr_t ReallocBuffer(std::uintptr_t buf,
                                 std::size_t new_size,
                                 bool priv,
                                 const char* label) {
        if (new_size == 0) {
            DestroyBuffer(buf);
            return 0;
        }
        if (buf != 0) {
            auto it = buffer_sizes_.find(buf);
            if (it != buffer_sizes_.end() && it->second >= new_size) {
                if (label != nullptr && label[0] != '\0') {
                    id<MTLBuffer> existing =
                            (__bridge id<MTLBuffer>)reinterpret_cast<void*>(
                                    buf);
                    [existing setLabel:[NSString stringWithUTF8String:label]];
                }
                return buf;
            }
        }
        DestroyBuffer(buf);
        return priv ? CreatePrivateBuffer(new_size, label)
                    : CreateBuffer(new_size, label);
    }

    void ResetPassState() {
        current_program_ = -1;
        for (auto& b : buffer_bindings_) b = {};
        for (auto& t : texture_bindings_) t = {};
        for (auto& s : sampler_bindings_) s = {};
    }

    void SetBufferBinding(std::uint32_t binding,
                          id<MTLBuffer> buffer,
                          NSUInteger offset) {
        if (binding < buffer_bindings_.size()) {
            buffer_bindings_[binding] = {buffer, offset, true};
        }
        if (encoder_) {
            [encoder_ setBuffer:buffer offset:offset atIndex:binding];
        }
    }

    void SetTextureBinding(std::uint32_t binding, id<MTLTexture> texture) {
        if (binding < texture_bindings_.size()) {
            texture_bindings_[binding] = {texture, true};
        }
        if (encoder_) {
            [encoder_ setTexture:texture atIndex:binding];
        }
    }

    void SetSamplerBinding(std::uint32_t binding,
                           id<MTLSamplerState> sampler_state) {
        if (binding < sampler_bindings_.size()) {
            sampler_bindings_[binding] = {sampler_state, true};
        }
        if (encoder_) {
            [encoder_ setSamplerState:sampler_state atIndex:binding];
        }
    }

    void ApplyEncoderState() {
        if (!encoder_) {
            return;
        }
        if (current_program_ >= 0 &&
            current_program_ < static_cast<int>(ComputeProgramId::kCount)) {
            [encoder_ setComputePipelineState:pipelines_[current_program_]];
        }
        for (std::size_t i = 0; i < buffer_bindings_.size(); ++i) {
            const auto& b = buffer_bindings_[i];
            if (b.is_bound) {
                [encoder_ setBuffer:b.buffer
                             offset:b.offset
                            atIndex:static_cast<NSUInteger>(i)];
            }
        }
        for (std::size_t i = 0; i < texture_bindings_.size(); ++i) {
            const auto& t = texture_bindings_[i];
            if (t.is_bound) {
                [encoder_ setTexture:t.texture
                             atIndex:static_cast<NSUInteger>(i)];
            }
        }
        for (std::size_t i = 0; i < sampler_bindings_.size(); ++i) {
            const auto& s = sampler_bindings_[i];
            if (s.is_bound) {
                [encoder_ setSamplerState:s.sampler
                                  atIndex:static_cast<NSUInteger>(i)];
            }
        }
    }

    void RestartComputeEncoder(id<MTLCommandBuffer> cb, bool restore_state) {
        if (!cb) {
            return;
        }
        if (encoder_) {
            [encoder_ endEncoding];
        }
        encoder_ = [cb computeCommandEncoder];
        if (restore_state) {
            ApplyEncoderState();
        }
    }

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> library_ = nil;
    id<MTLComputePipelineState>
            pipelines_[static_cast<int>(ComputeProgramId::kCount)] = {};
    id<MTLSamplerState> sampler_ = nil;

    id<MTLCommandBuffer> geom_cb_ = nil;
    id<MTLCommandBuffer> comp_cb_ = nil;
    // Tracks the most recently committed composite CB so the destructor can
    // wait for its completion handler to finish before freeing C++ state.
    id<MTLCommandBuffer> last_submitted_comp_cb_ = nil;
    id<MTLComputeCommandEncoder> encoder_ = nil;

    int current_program_ = -1;

    std::array<BufferBindingState, kMaxBindings> buffer_bindings_;
    std::array<TextureBindingState, kMaxBindings> texture_bindings_;
    std::array<SamplerBindingState, kMaxBindings> sampler_bindings_;

    bool last_submit_succeeded_ = true;
    bool onesweep_programs_valid_ = false;

    std::unordered_map<std::uintptr_t, std::size_t> buffer_sizes_;
    std::unordered_map<std::uintptr_t, std::pair<std::uint32_t, std::uint32_t>>
            texture_sizes_;
};

std::unique_ptr<GaussianSplatGpuContext> CreateComputeGpuContextMetal(
        std::uintptr_t device_handle, std::uintptr_t command_queue_handle) {
    return std::make_unique<GaussianSplatGpuContextMetal>(device_handle,
                                                          command_queue_handle);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // defined(__APPLE__)
