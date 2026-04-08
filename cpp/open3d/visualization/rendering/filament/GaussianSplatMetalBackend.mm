// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/rendering/filament/GaussianSplatRenderer.h"

#if defined(__APPLE__)

#include <memory>
#include <unordered_map>

#include "open3d/utility/Logging.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentNativeInterop.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"
#include "open3d/visualization/rendering/filament/GaussianSplatDataPacking.h"
#include "open3d/visualization/rendering/filament/ComputeGPU.h"
#include "open3d/visualization/rendering/filament/GaussianSplatPassRunner.h"

namespace open3d {
namespace visualization {
namespace rendering {

class GaussianSplatMetalBackend final : public GaussianSplatRenderer::Backend {
public:
    GaussianSplatMetalBackend(
            FilamentResourceManager& resource_mgr,
            const GaussianSplatRenderer::RenderConfig& config)
        : config_(config) {
        (void)resource_mgr;
        FilamentMetalNativeHandles mh =
                GetFilamentMetalNativeHandles(EngineInstance::GetPlatform());
        if (mh.valid) {
            gpu_ = CreateComputeGpuContextMetal(mh.device,
                                                mh.command_queue);
        }
    }

    ~GaussianSplatMetalBackend() override { Cleanup(); }

    const char* GetName() const override { return "Metal"; }

    void BeginFrame(std::uint64_t) override {}

    void ForgetView(const FilamentView& view) override {
        auto it = view_states_.find(&view);
        if (it != view_states_.end()) {
            DestroyViewState(it->second);
            view_states_.erase(it);
        }
    }

    bool RenderGeometryStage(
            const FilamentView& view,
            const FilamentScene& scene,
            const GaussianSplatRenderer::ViewRenderData& render_data,
            const std::vector<GaussianSplatRenderer::PassDispatch>&
                    dispatches,
            GaussianSplatRenderer::OutputTargets& targets) override {
        if (!gpu_) {
            return false;
        }

        const GaussianSplatSourceData* source =
                scene.GetGaussianSplatSourceData();
        if (!source || source->splat_count == 0) {
            return false;
        }

        // Pack only the view-params UBO (208 bytes) — cheap every frame.
        PackedGaussianScene packed =
                PackGaussianViewParams(*source, render_data, config_);
        if (!packed.valid) {
            return false;
        }

        auto& vs = view_states_[&view];

        const std::uint64_t scene_id = scene.GetGeometryChangeId();
        const bool scene_changed =
                (scene_id != vs.cached_scene_id ||
                 source->splat_count != vs.cached_splat_count);

        // Pack the large per-splat arrays (N * ~160 B) only when geometry
        // changes; on camera-move frames these stay empty and the upload
        // in RunGaussianGeometryPasses is skipped.
        if (scene_changed) {
            PackGaussianSceneAttributes(*source, config_, packed);
        }

        return RunGaussianGeometryPasses(*gpu_, config_, packed, dispatches, vs,
                                        scene_id, source->splat_count,
                                        scene_changed);
    }

    bool RenderCompositeStage(
            const FilamentView& view,
            const GaussianSplatRenderer::ViewRenderData&,
            const std::vector<GaussianSplatRenderer::PassDispatch>&
                    dispatches,
            GaussianSplatRenderer::OutputTargets& targets) override {
        if (!gpu_) {
            return false;
        }
        auto it = view_states_.find(&view);
        if (it == view_states_.end() || it->second.view_params_buf == 0) {
            return false;
        }
        return RunGaussianCompositePass(*gpu_, config_, dispatches, it->second,
                                       targets);
    }

private:
    void DestroyViewState(GaussianSplatViewGpuResources& vs) {
        if (!gpu_) {
            return;
        }
        auto destroy_buf = [&](std::uintptr_t& b) {
            if (b != 0) {
                gpu_->DestroyBuffer(b);
                b = 0;
            }
        };
        destroy_buf(vs.view_params_buf);
        destroy_buf(vs.positions_buf);
        destroy_buf(vs.scales_buf);
        destroy_buf(vs.rotations_buf);
        destroy_buf(vs.dc_opacity_buf);
        destroy_buf(vs.sh_buf);
        destroy_buf(vs.projected_buf);
        destroy_buf(vs.tile_counts_buf);
        destroy_buf(vs.tile_offsets_buf);
        destroy_buf(vs.tile_heads_buf);
        destroy_buf(vs.counters_buf);
        destroy_buf(vs.tile_entries_buf);
        destroy_buf(vs.dispatch_args_buf);
        destroy_buf(vs.sort_keys_buf[0]);
        destroy_buf(vs.sort_keys_buf[1]);
        destroy_buf(vs.sort_values_buf[0]);
        destroy_buf(vs.sort_values_buf[1]);
        destroy_buf(vs.histogram_buf);
        destroy_buf(vs.radix_params_buf);
        destroy_buf(vs.sorted_entries_buf);
        if (vs.composite_depth_tex != 0) {
            gpu_->DestroyTexture(vs.composite_depth_tex);
            vs.composite_depth_tex = 0;
        }
    }

    void Cleanup() {
        for (auto& pair : view_states_) {
            DestroyViewState(pair.second);
        }
        view_states_.clear();
        gpu_.reset();
    }

    const GaussianSplatRenderer::RenderConfig& config_;
    std::unordered_map<const FilamentView*, GaussianSplatViewGpuResources>
            view_states_;
    std::unique_ptr<GaussianSplatGpuContext> gpu_;
};

std::unique_ptr<GaussianSplatRenderer::Backend> CreateGaussianSplatMetalBackend(
        FilamentResourceManager& resource_mgr,
        const GaussianSplatRenderer::RenderConfig& config) {
    return std::make_unique<GaussianSplatMetalBackend>(resource_mgr, config);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d

#endif  // defined(__APPLE__)
