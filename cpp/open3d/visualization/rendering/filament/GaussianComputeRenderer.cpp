// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianComputeRenderer.h"

#include "open3d/utility/FileSystem.h"
#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentNativeInterop.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"
#include "open3d/visualization/rendering/filament/GaussianComputeMetalShaders.h"

namespace open3d {
namespace visualization {
namespace rendering {

class GaussianComputeRenderer::Backend {
public:
    virtual ~Backend() = default;

    virtual const char* GetName() const = 0;
    virtual void BeginFrame(std::uint64_t frame_index) = 0;
    virtual void ForgetView(const FilamentView& view) = 0;
    virtual bool Render(const FilamentView& view,
                        const FilamentScene& scene,
                        const ViewRenderData& render_data,
                        const std::vector<PassDispatch>& dispatches,
                        OutputTargets& targets) = 0;
};

namespace {

#if defined(OPEN3D_BUILD_GAUSSIAN_SPLAT_COMPUTE)
struct LoadedShaderSource {
    std::string path;
    std::string source;
    bool loaded = false;
};

struct LoadedShaderBinary {
    std::string path;
    std::vector<char> bytes;
    bool loaded = false;
};

bool ReadTextFile(const std::string& path, std::string* contents) {
    std::vector<char> bytes;
    std::string error;
    if (!utility::filesystem::FReadToBuffer(path, bytes, &error)) {
        utility::LogWarning("Failed to read Gaussian compute shader {}: {}",
                            path, error);
        return false;
    }

    contents->assign(bytes.begin(), bytes.end());
    return true;
}

bool ReadBinaryFile(const std::string& path, std::vector<char>* bytes) {
    std::string error;
    if (!utility::filesystem::FReadToBuffer(path, *bytes, &error)) {
        utility::LogWarning("Failed to read Gaussian compute shader {}: {}",
                            path, error);
        return false;
    }
    return true;
}

class GaussianComputePlaceholderBackend final
    : public GaussianComputeRenderer::Backend {
public:
    explicit GaussianComputePlaceholderBackend(const char* name)
        : name_(name) {}

    const char* GetName() const override { return name_; }

    void BeginFrame(std::uint64_t) override {}

    void ForgetView(const FilamentView& view) override {
        logged_views_.erase(&view);
    }

    bool Render(const FilamentView& view,
                const FilamentScene&,
                const GaussianComputeRenderer::ViewRenderData&,
                const std::vector<GaussianComputeRenderer::PassDispatch>&,
                GaussianComputeRenderer::OutputTargets&) override {
        if (logged_views_.insert(&view).second) {
            utility::LogInfo(
                    "Gaussian compute backend '{}' is selected but GPU "
                    "dispatch is not implemented yet.",
                    name_);
        }
        return false;
    }

private:
    const char* name_;
    std::unordered_set<const FilamentView*> logged_views_;
};

class GaussianComputeVulkanBackend final
    : public GaussianComputeRenderer::Backend {
public:
    const char* GetName() const override { return "Vulkan"; }

    void BeginFrame(std::uint64_t) override {}

    void ForgetView(const FilamentView&) override {}

    bool Render(const FilamentView& view,
                const FilamentScene&,
                const GaussianComputeRenderer::ViewRenderData&,
                const std::vector<GaussianComputeRenderer::PassDispatch>&,
                GaussianComputeRenderer::OutputTargets&) override {
        if (logged_views_.find(&view) != logged_views_.end()) {
            return false;
        }

        logged_views_.insert(&view);
        const FilamentVulkanNativeHandles handles =
                GetFilamentVulkanNativeHandles(EngineInstance::GetPlatform());
        if (!handles.valid) {
            utility::LogWarning(
                    "Gaussian compute Vulkan backend could not query Filament "
                    "native Vulkan handles.");
            return false;
        }

        EnsureSpirvShadersLoaded();

        std::size_t loaded_shader_count = 0;
        for (const auto& shader : spirv_shaders_) {
            if (shader.second.loaded) {
                ++loaded_shader_count;
            }
        }

        utility::LogInfo(
                "Gaussian compute Vulkan backend staged {} of {} SPIR-V passes "
                "on Filament device={} queue={} family={} index={}; Vulkan "
                "pipeline creation remains pending.",
                loaded_shader_count, spirv_shaders_.size(), handles.device,
                handles.graphics_queue, handles.graphics_queue_family_index,
                handles.graphics_queue_index);
        return false;
    }

private:
    void EnsureSpirvShadersLoaded() {
        if (!spirv_shaders_.empty()) {
            return;
        }

        static const std::pair<GaussianComputeRenderer::PassType, const char*>
                kPassFiles[] = {
                        {GaussianComputeRenderer::PassType::kProjection,
                         "gaussian_project.spv"},
                        {GaussianComputeRenderer::PassType::kTilePrefixSum,
                         "gaussian_prefix_sum.spv"},
                        {GaussianComputeRenderer::PassType::kTileScatter,
                         "gaussian_scatter.spv"},
                        {GaussianComputeRenderer::PassType::kTileSort,
                         "gaussian_sort.spv"},
                        {GaussianComputeRenderer::PassType::kComposite,
                         "gaussian_composite.spv"},
                };

        const std::string shader_root =
                EngineInstance::GetResourcePath() + "/gaussian_compute/";
        for (const auto& pass : kPassFiles) {
            LoadedShaderBinary shader;
            shader.path = shader_root + pass.second;
            shader.loaded = ReadBinaryFile(shader.path, &shader.bytes);
            spirv_shaders_.emplace(pass.first, std::move(shader));
        }
    }

    std::unordered_set<const FilamentView*> logged_views_;
    std::unordered_map<GaussianComputeRenderer::PassType, LoadedShaderBinary>
            spirv_shaders_;
};

class GaussianComputeMetalBackend final
    : public GaussianComputeRenderer::Backend {
public:
    ~GaussianComputeMetalBackend() override {
        for (auto& entry : pipelines_) {
            DestroyMetalComputePipeline(entry.second.pipeline);
        }
    }

    const char* GetName() const override { return "Metal"; }

    void BeginFrame(std::uint64_t) override {}

    void ForgetView(const FilamentView&) override {}

    bool
    Render(const FilamentView& view,
           const FilamentScene& scene,
           const GaussianComputeRenderer::ViewRenderData& render_data,
           const GaussianComputeRenderer::RenderConfig& config,
           const std::vector<GaussianComputeRenderer::PassDispatch>& dispatches,
           GaussianComputeRenderer::OutputTargets& targets) override {
        const FilamentMetalNativeHandles handles =
                GetFilamentMetalNativeHandles(EngineInstance::GetPlatform());
        if (!handles.valid) {
            utility::LogWarning(
                    "Gaussian compute Metal backend could not query Filament "
                    "native Metal handles.");
            return false;
        }

        EnsurePassPipelinesLoaded(handles.device);
        PackedGaussianScene packed =
                PackGaussianSceneInputs(scene, render_data, config, dispatches);
        if (!packed.valid) {
            return false;
        }

        std::vector<MetalBufferHandle> buffers_to_destroy;
        auto destroy_buffers = [&buffers_to_destroy]() {
            for (MetalBufferHandle handle : buffers_to_destroy) {
                DestroyMetalSharedBuffer(handle);
            }
        };

        auto create_buffer = [&](const std::string& label, std::size_t size,
                                 const void* initial_data) {
            std::string error;
            MetalBufferHandle handle = CreateMetalSharedBuffer(
                    handles.device, std::max<std::size_t>(size, 16u), label,
                    &error);
            if (!handle.valid) {
                utility::LogWarning(
                        "Gaussian compute Metal backend failed to allocate {}: "
                        "{}",
                        label, error);
                return handle;
            }
            buffers_to_destroy.push_back(handle);
            if (size > 0) {
                if (initial_data) {
                    if (!UploadMetalSharedBuffer(handle, initial_data, size, 0,
                                                 &error)) {
                        utility::LogWarning(
                                "Gaussian compute Metal backend failed to "
                                "upload {}: {}",
                                label, error);
                        return MetalBufferHandle();
                    }
                } else {
                    std::vector<std::uint8_t> zeros(size, 0u);
                    if (!UploadMetalSharedBuffer(handle, zeros.data(), size, 0,
                                                 &error)) {
                        utility::LogWarning(
                                "Gaussian compute Metal backend failed to "
                                "clear {}: {}",
                                label, error);
                        return MetalBufferHandle();
                    }
                }
            }
            return handle;
        };

        const std::size_t projected_size =
                packed.splat_count * sizeof(PackedProjectedGaussian);
        const std::size_t tile_scalar_size =
                packed.tile_count * sizeof(std::uint32_t);

        MetalBufferHandle view_params_buffer = create_buffer(
                "gaussian_view_params", sizeof(PackedGaussianViewParams),
                &packed.view_params);
        MetalBufferHandle positions_buffer =
                create_buffer("gaussian_positions",
                              packed.positions.size() * sizeof(Std430Vec4),
                              packed.positions.data());
        MetalBufferHandle scales_buffer =
                create_buffer("gaussian_scales",
                              packed.log_scales.size() * sizeof(Std430Vec4),
                              packed.log_scales.data());
        MetalBufferHandle rotations_buffer =
                create_buffer("gaussian_rotations",
                              packed.rotations.size() * sizeof(Std430Vec4),
                              packed.rotations.data());
        MetalBufferHandle dc_opacity_buffer =
                create_buffer("gaussian_dc_opacity",
                              packed.dc_opacity.size() * sizeof(Std430Vec4),
                              packed.dc_opacity.data());
        MetalBufferHandle sh_buffer = create_buffer(
                "gaussian_sh",
                packed.sh_coefficients.size() * sizeof(Std430Vec4),
                packed.sh_coefficients.data());
        MetalBufferHandle projected_buffer =
                create_buffer("gaussian_projected", projected_size, nullptr);
        MetalBufferHandle tile_counts_buffer = create_buffer(
                "gaussian_tile_counts", tile_scalar_size, nullptr);
        MetalBufferHandle tile_offsets_buffer = create_buffer(
                "gaussian_tile_offsets", tile_scalar_size, nullptr);
        MetalBufferHandle tile_heads_buffer =
                create_buffer("gaussian_tile_heads", tile_scalar_size, nullptr);
        std::array<std::uint32_t, kGaussianCounterCount> counters = {0u, 0u, 0u,
                                                                     0u};
        MetalBufferHandle counters_buffer = create_buffer(
                "gaussian_counters", sizeof(counters), counters.data());

        if (!view_params_buffer.valid || !positions_buffer.valid ||
            !scales_buffer.valid || !rotations_buffer.valid ||
            !dc_opacity_buffer.valid || !sh_buffer.valid ||
            !projected_buffer.valid || !tile_counts_buffer.valid ||
            !tile_offsets_buffer.valid || !tile_heads_buffer.valid ||
            !counters_buffer.valid) {
            destroy_buffers();
            return false;
        }

        auto find_dispatch = [&](GaussianComputeRenderer::PassType type)
                -> const GaussianComputeRenderer::PassDispatch* {
            for (const auto& dispatch : dispatches) {
                if (dispatch.type == type) {
                    return &dispatch;
                }
            }
            return nullptr;
        };

        auto dispatch_pass = [&](GaussianComputeRenderer::PassType type,
                                 std::initializer_list<MetalBufferBinding>
                                         bindings) {
            const auto* dispatch = find_dispatch(type);
            auto pipeline_it = pipelines_.find(type);
            if (!dispatch || pipeline_it == pipelines_.end() ||
                !pipeline_it->second.pipeline.valid) {
                return false;
            }

            MetalComputeDispatch metal_dispatch;
            metal_dispatch.pipeline = pipeline_it->second.pipeline;
            metal_dispatch.buffers.assign(bindings.begin(), bindings.end());
            metal_dispatch.group_count_x =
                    std::max(1, dispatch->group_count.x());
            metal_dispatch.group_count_y =
                    std::max(1, dispatch->group_count.y());
            metal_dispatch.group_count_z =
                    std::max(1, dispatch->group_count.z());
            metal_dispatch.thread_count_x =
                    std::max(1, dispatch->group_size.x());
            metal_dispatch.thread_count_y =
                    std::max(1, dispatch->group_size.y());
            metal_dispatch.thread_count_z =
                    std::max(1, dispatch->group_size.z());

            std::string error;
            if (!DispatchMetalComputePipelines(handles.command_queue,
                                               {metal_dispatch}, &error)) {
                utility::LogWarning(
                        "Gaussian compute Metal backend failed in pass {}: {}",
                        pipeline_it->second.entry_point, error);
                return false;
            }
            return true;
        };

        if (!dispatch_pass(GaussianComputeRenderer::PassType::kProjection,
                           {{0u, view_params_buffer, 0u},
                            {1u, positions_buffer, 0u},
                            {2u, scales_buffer, 0u},
                            {3u, rotations_buffer, 0u},
                            {4u, dc_opacity_buffer, 0u},
                            {5u, sh_buffer, 0u},
                            {6u, projected_buffer, 0u}}) ||
            !dispatch_pass(GaussianComputeRenderer::PassType::kTilePrefixSum,
                           {{0u, view_params_buffer, 0u},
                            {6u, projected_buffer, 0u},
                            {7u, tile_counts_buffer, 0u},
                            {8u, tile_offsets_buffer, 0u},
                            {9u, tile_heads_buffer, 0u},
                            {10u, counters_buffer, 0u}})) {
            destroy_buffers();
            return false;
        }

        std::string counter_error;
        if (!DownloadMetalSharedBuffer(counters_buffer, counters.data(),
                                       sizeof(counters), 0u, &counter_error)) {
            utility::LogWarning(
                    "Gaussian compute Metal backend failed to read counters: "
                    "{}",
                    counter_error);
            destroy_buffers();
            return false;
        }

        const std::uint32_t total_entries = counters[0];
        MetalBufferHandle tile_entries_buffer = create_buffer(
                "gaussian_tile_entries",
                std::max<std::size_t>(sizeof(PackedTileEntry),
                                      total_entries * sizeof(PackedTileEntry)),
                nullptr);
        MetalBufferHandle out_color_buffer = create_buffer(
                "gaussian_out_color",
                std::max<std::size_t>(sizeof(Std430Vec4),
                                      packed.pixel_count * sizeof(Std430Vec4)),
                nullptr);
        MetalBufferHandle out_depth_buffer = create_buffer(
                "gaussian_out_depth",
                std::max<std::size_t>(sizeof(float),
                                      packed.pixel_count * sizeof(float)),
                nullptr);
        if (!tile_entries_buffer.valid || !out_color_buffer.valid ||
            !out_depth_buffer.valid) {
            destroy_buffers();
            return false;
        }

        if (!dispatch_pass(GaussianComputeRenderer::PassType::kTileScatter,
                           {{0u, view_params_buffer, 0u},
                            {6u, projected_buffer, 0u},
                            {8u, tile_offsets_buffer, 0u},
                            {9u, tile_heads_buffer, 0u},
                            {11u, tile_entries_buffer, 0u}}) ||
            !dispatch_pass(GaussianComputeRenderer::PassType::kTileSort,
                           {{0u, view_params_buffer, 0u},
                            {7u, tile_counts_buffer, 0u},
                            {8u, tile_offsets_buffer, 0u},
                            {11u, tile_entries_buffer, 0u}}) ||
            !dispatch_pass(GaussianComputeRenderer::PassType::kComposite,
                           {{0u, view_params_buffer, 0u},
                            {6u, projected_buffer, 0u},
                            {7u, tile_counts_buffer, 0u},
                            {8u, tile_offsets_buffer, 0u},
                            {11u, tile_entries_buffer, 0u},
                            {12u, out_color_buffer, 0u},
                            {13u, out_depth_buffer, 0u}})) {
            destroy_buffers();
            return false;
        }

        std::vector<Std430Vec4> color_pixels(packed.pixel_count);
        std::vector<float> depth_pixels(packed.pixel_count, 0.0f);
        std::string readback_error;
        if (!DownloadMetalSharedBuffer(out_color_buffer, color_pixels.data(),
                                       color_pixels.size() * sizeof(Std430Vec4),
                                       0u, &readback_error) ||
            !DownloadMetalSharedBuffer(out_depth_buffer, depth_pixels.data(),
                                       depth_pixels.size() * sizeof(float), 0u,
                                       &readback_error)) {
            utility::LogWarning(
                    "Gaussian compute Metal backend failed to read outputs: {}",
                    readback_error);
            destroy_buffers();
            return false;
        }

        destroy_buffers();
        return UploadOutputTextures(resource_mgr_, color_pixels, depth_pixels,
                                    targets.width, targets.height, targets);
    }

private:
    struct CompiledPass {
        LoadedShaderSource source;
        MetalComputePipelineHandle pipeline;
        std::string entry_point;
    };

    void EnsurePassPipelinesLoaded(std::uintptr_t device_handle) {
        if (!pipelines_.empty()) {
            return;
        }

        static const struct {
            GaussianComputeRenderer::PassType type;
            const char* file_name;
            const char* entry_point;
        } kPassFiles[] = {
                {GaussianComputeRenderer::PassType::kProjection,
                 "gaussian_project.metal", "gaussian_project_main"},
                {GaussianComputeRenderer::PassType::kTilePrefixSum,
                 "gaussian_prefix_sum.metal", "gaussian_prefix_sum_main"},
                {GaussianComputeRenderer::PassType::kTileScatter,
                 "gaussian_scatter.metal", "gaussian_scatter_main"},
                {GaussianComputeRenderer::PassType::kTileSort,
                 "gaussian_sort.metal", "gaussian_sort_main"},
                {GaussianComputeRenderer::PassType::kComposite,
                 "gaussian_composite.metal", "gaussian_composite_main"},
        };

        const std::string shader_root =
                EngineInstance::GetResourcePath() + "/gaussian_compute/";
        for (const auto& pass : kPassFiles) {
            CompiledPass compiled_pass;
            compiled_pass.source.path = shader_root + pass.file_name;
            compiled_pass.entry_point = pass.entry_point;
            compiled_pass.source.loaded = ReadTextFile(
                    compiled_pass.source.path, &compiled_pass.source.source);
            if (compiled_pass.source.loaded) {
                std::string error;
                compiled_pass.pipeline = CompileMetalComputePipeline(
                        device_handle, compiled_pass.source.source,
                        compiled_pass.entry_point, pass.file_name, &error);
                if (!compiled_pass.pipeline.valid) {
                    utility::LogWarning(
                            "Failed to compile Gaussian Metal compute pass {}: "
                            "{}",
                            pass.file_name, error);
                }
            }
            pipelines_.emplace(pass.type, std::move(compiled_pass));
        }
    }

    std::unordered_map<GaussianComputeRenderer::PassType, CompiledPass>
            pipelines_;
};

std::unique_ptr<GaussianComputeRenderer::Backend> CreateBackend(
        RenderingType backend) {
    switch (backend) {
        case RenderingType::kVulkan:
            return std::unique_ptr<GaussianComputeRenderer::Backend>(
                    new GaussianComputeVulkanBackend());
        case RenderingType::kMetal:
            return std::unique_ptr<GaussianComputeRenderer::Backend>(
                    new GaussianComputeMetalBackend());
        case RenderingType::kDefault:
            return std::unique_ptr<GaussianComputeRenderer::Backend>(
                    new GaussianComputePlaceholderBackend("Default"));
        case RenderingType::kOpenGL:
            break;
    }
    return nullptr;
}
#endif

std::vector<GaussianComputeRenderer::PassDefinition> CreateDefaultPasses() {
    return {
            {GaussianComputeRenderer::PassType::kProjection,
             "Gaussian Projection", "gaussian_project.comp"},
            {GaussianComputeRenderer::PassType::kTilePrefixSum,
             "Gaussian Tile Prefix Sum", "gaussian_prefix_sum.comp"},
            {GaussianComputeRenderer::PassType::kTileScatter,
             "Gaussian Tile Scatter", "gaussian_scatter.comp"},
            {GaussianComputeRenderer::PassType::kTileSort, "Gaussian Tile Sort",
             "gaussian_sort.comp"},
            {GaussianComputeRenderer::PassType::kComposite,
             "Gaussian Composite", "gaussian_composite.comp"},
    };
}

int CeilDiv(int value, int divisor) { return (value + divisor - 1) / divisor; }

#if defined(OPEN3D_BUILD_GAUSSIAN_SPLAT_COMPUTE)
bool GaussianComputeBackendSupported(RenderingType backend) {
    if (!EngineInstance::GetPlatform()) {
        return false;
    }

    switch (backend) {
        case RenderingType::kDefault:
        case RenderingType::kVulkan:
        case RenderingType::kMetal:
            return true;
        case RenderingType::kOpenGL:
            return false;
    }
}
#endif  // OPEN3D_BUILD_GAUSSIAN_SPLAT_COMPUTE

}  // namespace

bool ProjectionInfoEquals(const Camera::ProjectionInfo& left,
                          const Camera::ProjectionInfo& right) {
    if (left.is_ortho != right.is_ortho ||
        left.is_intrinsic != right.is_intrinsic) {
        return false;
    }

    if (left.is_intrinsic) {
        return left.proj.intrinsics.fx == right.proj.intrinsics.fx &&
               left.proj.intrinsics.fy == right.proj.intrinsics.fy &&
               left.proj.intrinsics.cx == right.proj.intrinsics.cx &&
               left.proj.intrinsics.cy == right.proj.intrinsics.cy &&
               left.proj.intrinsics.near_plane ==
                       right.proj.intrinsics.near_plane &&
               left.proj.intrinsics.far_plane ==
                       right.proj.intrinsics.far_plane &&
               left.proj.intrinsics.width == right.proj.intrinsics.width &&
               left.proj.intrinsics.height == right.proj.intrinsics.height;
    }

    if (left.is_ortho) {
        return left.proj.ortho.projection == right.proj.ortho.projection &&
               left.proj.ortho.left == right.proj.ortho.left &&
               left.proj.ortho.right == right.proj.ortho.right &&
               left.proj.ortho.bottom == right.proj.ortho.bottom &&
               left.proj.ortho.top == right.proj.ortho.top &&
               left.proj.ortho.near_plane == right.proj.ortho.near_plane &&
               left.proj.ortho.far_plane == right.proj.ortho.far_plane;
    }

    return left.proj.perspective.fov_type == right.proj.perspective.fov_type &&
           left.proj.perspective.fov == right.proj.perspective.fov &&
           left.proj.perspective.aspect == right.proj.perspective.aspect &&
           left.proj.perspective.near_plane ==
                   right.proj.perspective.near_plane &&
           left.proj.perspective.far_plane == right.proj.perspective.far_plane;
}

bool ViewRenderDataEquals(
        const GaussianComputeRenderer::ViewRenderData& left,
        const GaussianComputeRenderer::ViewRenderData& right) {
    return left.viewport_origin == right.viewport_origin &&
           left.viewport_size == right.viewport_size &&
           left.camera_position.isApprox(right.camera_position) &&
           left.model_matrix.matrix().isApprox(right.model_matrix.matrix()) &&
           left.view_matrix.matrix().isApprox(right.view_matrix.matrix()) &&
           left.projection_matrix.matrix().isApprox(
                   right.projection_matrix.matrix()) &&
           left.culling_projection_matrix.matrix().isApprox(
                   right.culling_projection_matrix.matrix()) &&
           left.near_plane == right.near_plane &&
           left.far_plane == right.far_plane &&
           ProjectionInfoEquals(left.projection, right.projection);
}

bool PassDispatchEquals(const GaussianComputeRenderer::PassDispatch& left,
                        const GaussianComputeRenderer::PassDispatch& right) {
    return left.type == right.type && left.group_size == right.group_size &&
           left.group_count == right.group_count &&
           left.tile_count == right.tile_count;
}

bool PassDispatchesEqual(
        const std::vector<GaussianComputeRenderer::PassDispatch>& left,
        const std::vector<GaussianComputeRenderer::PassDispatch>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (size_t index = 0; index < left.size(); ++index) {
        if (!PassDispatchEquals(left[index], right[index])) {
            return false;
        }
    }
    return true;
}

GaussianComputeRenderer::GaussianComputeRenderer(
        filament::Engine& engine, FilamentResourceManager& resource_mgr)
    : engine_(engine),
      resource_mgr_(resource_mgr),
      pass_definitions_(CreateDefaultPasses()) {
#if defined(OPEN3D_BUILD_GAUSSIAN_SPLAT_COMPUTE)
    enabled_ =
            GaussianComputeBackendSupported(EngineInstance::GetBackendType());
    backend_ = CreateBackend(EngineInstance::GetBackendType());
#endif
}

GaussianComputeRenderer::~GaussianComputeRenderer() {
    for (auto& pair : outputs_) {
        ResetOutputTargets(pair.second);
    }
}

void GaussianComputeRenderer::BeginFrame() {
    ++frame_index_;
    if (backend_) {
        backend_->BeginFrame(frame_index_);
    }
}

void GaussianComputeRenderer::RenderView(FilamentView& view,
                                         const FilamentScene& scene) {
    if (!enabled_ || !scene.HasGaussianSplatGeometry()) {
        return;
    }

    ValidatePassShaderSources();

    auto& targets = PrepareOutputTargets(view);
    const std::uint64_t scene_change_id = scene.GetGeometryChangeId();
    const bool view_changed = UpdateViewRenderData(targets, view);
    const bool dispatch_changed = UpdatePassDispatches(targets, view);
    const bool scene_changed = targets.last_scene_change_id != scene_change_id;

    if (view_changed || dispatch_changed || scene_changed) {
        targets.needs_render = true;
    }

    if (!targets.needs_render) {
        return;
    }

    bool rendered = false;
    if (backend_ && targets.has_render_data) {
        rendered = backend_->Render(view, scene, targets.render_data,
                                    targets.pass_dispatches, targets);
    }

    targets.has_valid_output = rendered;
    targets.needs_render = false;
    targets.last_scene_change_id = scene_change_id;
    targets.last_updated_frame = frame_index_;
}

void GaussianComputeRenderer::PruneOutputs(
        const std::unordered_set<const FilamentView*>& live_views) {
    for (auto it = outputs_.begin(); it != outputs_.end();) {
        if (live_views.find(it->first) != live_views.end()) {
            ++it;
            continue;
        }

        if (backend_) {
            backend_->ForgetView(*it->first);
        }
        ResetOutputTargets(it->second);
        it = outputs_.erase(it);
    }
}

bool GaussianComputeRenderer::IsEnabled() const { return enabled_; }

void GaussianComputeRenderer::SetEnabled(bool enabled) {
    enabled_ = enabled && IsSupported();
}

bool GaussianComputeRenderer::IsSupported() const {
#if defined(OPEN3D_BUILD_GAUSSIAN_SPLAT_COMPUTE)
    return GaussianComputeBackendSupported(EngineInstance::GetBackendType());
#else
    return false;
#endif
}

bool GaussianComputeRenderer::HasOutput(const FilamentView& view) const {
    auto found = outputs_.find(&view);
    return found != outputs_.end() && found->second.color &&
           found->second.depth && found->second.render_target &&
           found->second.has_valid_output;
}

const GaussianComputeRenderer::ViewRenderData*
GaussianComputeRenderer::GetViewRenderData(const FilamentView& view) const {
    auto found = outputs_.find(&view);
    if (found == outputs_.end() || !found->second.has_render_data) {
        return nullptr;
    }
    return &found->second.render_data;
}

const std::vector<GaussianComputeRenderer::PassDispatch>*
GaussianComputeRenderer::GetPassDispatches(const FilamentView& view) const {
    auto found = outputs_.find(&view);
    if (found == outputs_.end()) {
        return nullptr;
    }
    return &found->second.pass_dispatches;
}

const std::vector<GaussianComputeRenderer::PassDefinition>&
GaussianComputeRenderer::GetPassDefinitions() const {
    return pass_definitions_;
}

std::string GaussianComputeRenderer::GetShaderSourcePath(
        PassType pass_type) const {
    const PassDefinition* pass = FindPassDefinition(pass_type);
    if (!pass) {
        return "";
    }

    return EngineInstance::GetResourcePath() + "/gaussian_compute/" +
           pass->shader_file;
}

bool GaussianComputeRenderer::HasShaderSource(PassType pass_type) const {
    const std::string shader_path = GetShaderSourcePath(pass_type);
    return !shader_path.empty() && utility::filesystem::FileExists(shader_path);
}

const GaussianComputeRenderer::RenderConfig&
GaussianComputeRenderer::GetRenderConfig() const {
    return render_config_;
}

void GaussianComputeRenderer::SetRenderConfig(const RenderConfig& config) {
    if (!ValidateRenderConfig(config)) {
        utility::LogWarning(
                "Ignoring invalid Gaussian compute render configuration.");
        return;
    }

    render_config_ = config;
    for (auto& pair : outputs_) {
        pair.second.pass_dispatches.clear();
        pair.second.has_valid_output = false;
        pair.second.needs_render = true;
    }
}

std::vector<GaussianComputeRenderer::PassDispatch>
GaussianComputeRenderer::BuildPassDispatches(const FilamentView& view) const {
    const ViewRenderData* render_data = GetViewRenderData(view);
    if (!render_data || render_data->viewport_size.x() <= 0 ||
        render_data->viewport_size.y() <= 0) {
        return {};
    }

    const Eigen::Vector2i tile_count(CeilDiv(render_data->viewport_size.x(),
                                             render_config_.tile_size.x()),
                                     CeilDiv(render_data->viewport_size.y(),
                                             render_config_.tile_size.y()));
    const int total_tiles = tile_count.x() * tile_count.y();

    return {
            {PassType::kProjection,
             Eigen::Vector3i(render_config_.projection_group_size, 1, 1),
             Eigen::Vector3i(
                     CeilDiv(total_tiles, render_config_.projection_group_size),
                     1, 1),
             tile_count},
            {PassType::kTilePrefixSum,
             Eigen::Vector3i(render_config_.prefix_sum_group_size, 1, 1),
             Eigen::Vector3i(
                     CeilDiv(total_tiles, render_config_.prefix_sum_group_size),
                     1, 1),
             tile_count},
            {PassType::kTileScatter,
             Eigen::Vector3i(render_config_.scatter_group_size, 1, 1),
             Eigen::Vector3i(
                     CeilDiv(total_tiles, render_config_.scatter_group_size), 1,
                     1),
             tile_count},
            {PassType::kTileSort,
             Eigen::Vector3i(render_config_.sort_group_size, 1, 1),
             Eigen::Vector3i(
                     CeilDiv(total_tiles, render_config_.sort_group_size), 1,
                     1),
             tile_count},
            {PassType::kComposite,
             Eigen::Vector3i(render_config_.composite_group_size.x(),
                             render_config_.composite_group_size.y(), 1),
             Eigen::Vector3i(CeilDiv(render_data->viewport_size.x(),
                                     render_config_.composite_group_size.x()),
                             CeilDiv(render_data->viewport_size.y(),
                                     render_config_.composite_group_size.y()),
                             1),
             tile_count},
    };
}

TextureHandle GaussianComputeRenderer::GetColorTexture(
        const FilamentView& view) const {
    auto found = outputs_.find(&view);
    return found != outputs_.end() ? found->second.color : TextureHandle();
}

TextureHandle GaussianComputeRenderer::GetDepthTexture(
        const FilamentView& view) const {
    auto found = outputs_.find(&view);
    return found != outputs_.end() ? found->second.depth : TextureHandle();
}

const char* GaussianComputeRenderer::GetBackendName() const {
    return backend_ ? backend_->GetName() : "Unavailable";
}

GaussianComputeRenderer::OutputTargets&
GaussianComputeRenderer::PrepareOutputTargets(const FilamentView& view) {
    auto viewport = view.GetViewport();
    auto width = static_cast<std::uint32_t>(viewport[2]);
    auto height = static_cast<std::uint32_t>(viewport[3]);

    auto& targets = outputs_[&view];
    if (targets.width == width && targets.height == height && targets.color &&
        targets.depth && targets.render_target) {
        return targets;
    }

    ResetOutputTargets(targets);

    targets.color =
            resource_mgr_.CreateColorAttachmentTexture(int(width), int(height));
    targets.depth =
            resource_mgr_.CreateDepthAttachmentTexture(int(width), int(height));
    targets.render_target =
            resource_mgr_.CreateRenderTarget(targets.color, targets.depth);
    targets.width = width;
    targets.height = height;
    targets.has_valid_output = false;
    targets.needs_render = true;
    return targets;
}

void GaussianComputeRenderer::ResetOutputTargets(OutputTargets& targets) {
    if (targets.render_target) {
        resource_mgr_.Destroy(targets.render_target);
        targets.render_target = RenderTargetHandle();
    }
    if (targets.color) {
        resource_mgr_.Destroy(targets.color);
        targets.color = TextureHandle();
    }
    if (targets.depth) {
        resource_mgr_.Destroy(targets.depth);
        targets.depth = TextureHandle();
    }
    targets.width = 0;
    targets.height = 0;
    targets.pass_dispatches.clear();
    targets.has_render_data = false;
    targets.has_valid_output = false;
    targets.needs_render = true;
    targets.last_scene_change_id = 0;
    targets.last_updated_frame = 0;
}

GaussianComputeRenderer::ViewRenderData
GaussianComputeRenderer::ExtractViewRenderData(const FilamentView& view) const {
    ViewRenderData data;

    auto viewport = view.GetViewport();
    data.viewport_origin = Eigen::Vector2i(viewport[0], viewport[1]);
    data.viewport_size = Eigen::Vector2i(viewport[2], viewport[3]);

    const auto* camera = view.GetCamera();
    if (camera) {
        data.camera_position = camera->GetPosition();
        data.model_matrix = camera->GetModelMatrix();
        data.view_matrix = camera->GetViewMatrix();
        data.projection_matrix = camera->GetProjectionMatrix();
        data.culling_projection_matrix = camera->GetCullingProjectionMatrix();
        data.projection = camera->GetProjection();
        data.near_plane = camera->GetNear();
        data.far_plane = camera->GetFar();
    }

    return data;
}

bool GaussianComputeRenderer::UpdateViewRenderData(OutputTargets& targets,
                                                   const FilamentView& view) {
    const ViewRenderData new_data = ExtractViewRenderData(view);
    if (!targets.has_render_data ||
        !ViewRenderDataEquals(targets.render_data, new_data)) {
        targets.render_data = new_data;
        targets.has_render_data = true;
        targets.needs_render = true;
        return true;
    }

    return false;
}

bool GaussianComputeRenderer::UpdatePassDispatches(OutputTargets& targets,
                                                   const FilamentView& view) {
    const std::vector<PassDispatch> new_dispatches = BuildPassDispatches(view);
    if (PassDispatchesEqual(targets.pass_dispatches, new_dispatches)) {
        return false;
    }

    targets.pass_dispatches = new_dispatches;
    targets.has_valid_output = false;
    return true;
}

const GaussianComputeRenderer::PassDefinition*
GaussianComputeRenderer::FindPassDefinition(PassType pass_type) const {
    for (const auto& pass : pass_definitions_) {
        if (pass.type == pass_type) {
            return &pass;
        }
    }
    return nullptr;
}

void GaussianComputeRenderer::ValidatePassShaderSources() const {
    if (pass_sources_validated_) {
        return;
    }
    pass_sources_validated_ = true;

    for (const auto& pass : pass_definitions_) {
        const std::string shader_path = GetShaderSourcePath(pass.type);
        if (!utility::filesystem::FileExists(shader_path)) {
            utility::LogWarning(
                    "Gaussian compute shader source for '{}' is missing at {}",
                    pass.debug_name, shader_path);
        }
    }
}

bool GaussianComputeRenderer::ValidateRenderConfig(
        const RenderConfig& config) const {
    return config.tile_size.x() > 0 && config.tile_size.y() > 0 &&
           config.projection_group_size > 0 &&
           config.prefix_sum_group_size > 0 && config.scatter_group_size > 0 &&
           config.sort_group_size > 0 && config.composite_group_size.x() > 0 &&
           config.composite_group_size.y() > 0 && config.max_sh_degree >= 0;
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d