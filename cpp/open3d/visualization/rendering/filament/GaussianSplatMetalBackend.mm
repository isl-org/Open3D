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
#include "open3d/visualization/rendering/filament/GaussianSplatOutputTargetsApple.h"
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

        const GaussianSplatPackedAttrs* attrs =
                scene.GetGaussianSplatPackedAttrs();
        if (!attrs || attrs->splat_count == 0) {
            return false;
        }

        // Pack only the view-params UBO (288 bytes) — cheap every frame.
        PackedGaussianScene frame =
                PackGaussianViewParams(*attrs, render_data, config_);
        if (!frame.valid) {
            return false;
        }

        auto& vs = view_states_[&view];

        const std::uint64_t scene_id = scene.GetGeometryChangeId();
        const bool scene_changed =
                (scene_id != vs.cached_scene_id ||
                 attrs->splat_count != vs.cached_splat_count);

        return RunGaussianGeometryPasses(*gpu_, config_, frame, *attrs,
                                        dispatches, vs, scene_id,
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

    bool PrepareOutputTextures(
            FilamentView& view,
            FilamentResourceManager& resource_mgr,
            std::uint32_t width,
            std::uint32_t height,
            GaussianSplatRenderer::OutputTargets& targets) override {
        return PrepareGaussianImportedRenderTargetsApple(
                view, resource_mgr, width, height, targets);
    }

    void ReleaseOutputTextures(
            FilamentResourceManager&,
            GaussianSplatRenderer::OutputTargets& targets) override {
        ReleaseGaussianImportedMTLTexturesApple(targets);
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
