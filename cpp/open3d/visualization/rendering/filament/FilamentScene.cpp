// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
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

// 4068: Filament has some clang-specific vectorizing pragma's that MSVC flags
// 4146: PixelBufferDescriptor assert unsigned is positive before subtracting
//       but MSVC can't figure that out.
// 4293: Filament's utils/algorithm.h utils::details::clz() does strange
//       things with MSVC. Somehow sizeof(unsigned int) > 4, but its size is
//       32 so that x >> 32 gives a warning. (Or maybe the compiler can't
//       determine the if statement does not run.)
// 4305: LightManager.h needs to specify some constants as floats
#include <unordered_set>
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068 4146 4293 4305)
#endif  // _MSC_VER

#include <backend/PixelBufferDescriptor.h>  // bogus 4146 warning on MSVC
#include <filament/Engine.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/MaterialInstance.h>
#include <filament/Renderer.h>
#include <filament/Scene.h>
#include <filament/Skybox.h>
#include <filament/SwapChain.h>
#include <filament/TextureSampler.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>
#include <filament/View.h>
#include <geometry/SurfaceOrientation.h>
#include <utils/EntityManager.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

// We do NOT include this first because it includes Image.h, which includes
// the fmt library, which includes windows.h (on Windows), which #defines
// OPAQUE (!!??) which causes syntax errors with filament/View.h which tries
// to make OPAQUE an member of a class enum. So include this after all the
// Filament headers to avoid this problem.
#if 1  // (enclose in #if so that apply-style doesn't move this)
#include "open3d/visualization/rendering/filament/FilamentScene.h"
#endif  // 1

#include "open3d/geometry/BoundingVolume.h"
#include "open3d/geometry/LineSet.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/utility/Console.h"
#include "open3d/visualization/rendering/Light.h"
#include "open3d/visualization/rendering/Material.h"
#include "open3d/visualization/rendering/Model.h"
#include "open3d/visualization/rendering/RendererHandle.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentEntitiesMods.h"
#include "open3d/visualization/rendering/filament/FilamentGeometryBuffersBuilder.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"

namespace {  // avoid polluting global namespace, since only used here
/// @cond

void DeallocateBuffer(void* buffer, size_t size, void* user_ptr) {
    free(buffer);
}

namespace defaults_mapping {

using GeometryType = open3d::geometry::Geometry::GeometryType;
using MaterialHandle = open3d::visualization::rendering::MaterialHandle;
using ResourceManager =
        open3d::visualization::rendering::FilamentResourceManager;

std::unordered_map<std::string, MaterialHandle> shader_mappings = {
        {"defaultLit", ResourceManager::kDefaultLit},
        {"defaultLitTransparency",
         ResourceManager::kDefaultLitWithTransparency},
        {"defaultUnlit", ResourceManager::kDefaultUnlit},
        {"normals", ResourceManager::kDefaultNormalShader},
        {"depth", ResourceManager::kDefaultDepthShader},
        {"unlitGradient", ResourceManager::kDefaultUnlitGradientShader},
        {"unlitSolidColor", ResourceManager::kDefaultUnlitSolidColorShader},
};

MaterialHandle kColorOnlyMesh = ResourceManager::kDefaultUnlit;
MaterialHandle kPlainMesh = ResourceManager::kDefaultLit;
MaterialHandle kMesh = ResourceManager::kDefaultLit;

MaterialHandle kColoredPointcloud = ResourceManager::kDefaultUnlit;
MaterialHandle kPointcloud = ResourceManager::kDefaultLit;

MaterialHandle kLineset = ResourceManager::kDefaultUnlit;

}  // namespace defaults_mapping

namespace converters {
using EigenMatrix =
        open3d::visualization::rendering::FilamentScene::Transform::MatrixType;
using FilamentMatrix = filament::math::mat4f;
EigenMatrix EigenMatrixFromFilamentMatrix(const filament::math::mat4f& fm) {
    EigenMatrix em;

    em << fm(0, 0), fm(0, 1), fm(0, 2), fm(0, 3), fm(1, 0), fm(1, 1), fm(1, 2),
            fm(1, 3), fm(2, 0), fm(2, 1), fm(2, 2), fm(2, 3), fm(3, 0),
            fm(3, 1), fm(3, 2), fm(3, 3);

    return em;
}

FilamentMatrix FilamentMatrixFromEigenMatrix(const EigenMatrix& em) {
    // Filament matrices is column major and Eigen's - row major
    return FilamentMatrix(FilamentMatrix::row_major_init{
            em(0, 0), em(0, 1), em(0, 2), em(0, 3), em(1, 0), em(1, 1),
            em(1, 2), em(1, 3), em(2, 0), em(2, 1), em(2, 2), em(2, 3),
            em(3, 0), em(3, 1), em(3, 2), em(3, 3)});
}
}  // namespace converters
/// @endcond
}  // namespace

namespace open3d {
namespace visualization {
namespace rendering {

FilamentScene::FilamentScene(filament::Engine& engine,
                             FilamentResourceManager& resource_mgr,
                             Renderer& renderer)
    : Scene(renderer), engine_(engine), resource_mgr_(resource_mgr) {
    scene_ = engine_.createScene();
    CreateSunDirectionalLight();
}

FilamentScene::~FilamentScene() {}

ViewHandle FilamentScene::AddView(std::int32_t x,
                                  std::int32_t y,
                                  std::uint32_t w,
                                  std::uint32_t h) {
    auto handle = ViewHandle::Next();
    auto view = std::make_unique<FilamentView>(engine_, *this, resource_mgr_);

    view->SetViewport(x, y, w, h);
    if (!views_.empty()) {
        view->SetDiscardBuffers(View::TargetBuffers::DepthAndStencil);
    }

    ViewContainer c;
    c.view = std::move(view);
    views_.emplace(handle, std::move(c));

    return handle;
}

View* FilamentScene::GetView(const ViewHandle& view_id) const {
    auto found = views_.find(view_id);
    if (found != views_.end()) {
        return found->second.view.get();
    }

    return nullptr;
}

void FilamentScene::SetViewActive(const ViewHandle& view_id, bool is_active) {
    auto found = views_.find(view_id);
    if (found != views_.end()) {
        found->second.is_active = is_active;
        found->second.render_count = -1;
    }
}

void FilamentScene::SetRenderOnce(const ViewHandle& view_id) {
    auto found = views_.find(view_id);
    if (found != views_.end()) {
        found->second.is_active = true;
        // NOTE: This value should match the value of render_count_ in
        // FilamentRenderer::EnableCaching
        found->second.render_count = 2;
    }
}

void FilamentScene::RemoveView(const ViewHandle& view_id) {
    views_.erase(view_id);
}

void FilamentScene::AddCamera(const std::string& camera_name,
                              std::shared_ptr<Camera> cam) {}

void FilamentScene::RemoveCamera(const std::string& camera_name) {}

void FilamentScene::SetActiveCamera(const std::string& camera_name) {}

MaterialInstanceHandle FilamentScene::AssignMaterialToFilamentGeometry(
        filament::RenderableManager::Builder& builder,
        const Material& material) {
    // TODO: put this in a method
    auto shader = defaults_mapping::shader_mappings[material.shader];
    if (!shader) shader = defaults_mapping::kColorOnlyMesh;

    auto material_instance = resource_mgr_.CreateMaterialInstance(shader);
    auto wmat_instance = resource_mgr_.GetMaterialInstance(material_instance);
    if (!wmat_instance.expired()) {
        builder.material(0, wmat_instance.lock().get());
    }
    return material_instance;
}

bool FilamentScene::AddGeometry(const std::string& object_name,
                                const geometry::Geometry3D& geometry,
                                const Material& material,
                                const std::string& downsampled_name /*= ""*/,
                                size_t downsample_threshold /*= SIZE_MAX*/) {
    if (geometries_.count(object_name) > 0) {
        utility::LogWarning(
                "Geometry {} has already been added to scene graph.",
                object_name);
        return false;
    }

    // Build Filament buffers
    auto geometry_buffer_builder = GeometryBuffersBuilder::GetBuilder(geometry);
    if (!geometry_buffer_builder) {
        utility::LogWarning("Geometry type {} is not supported yet!",
                            static_cast<size_t>(geometry.GetGeometryType()));
        return false;
    }

    auto buffer_builder = GeometryBuffersBuilder::GetBuilder(geometry);
    if (!downsampled_name.empty()) {
        buffer_builder->SetDownsampleThreshold(downsample_threshold);
    }
    auto buffers = buffer_builder->ConstructBuffers();
    auto vb = std::get<0>(buffers);
    auto ib = std::get<1>(buffers);
    auto ib_downsampled = std::get<2>(buffers);
    filament::Box aabb = buffer_builder->ComputeAABB();  // expensive
    bool success = CreateAndAddFilamentEntity(object_name, *buffer_builder,
                                              aabb, vb, ib, material);
    if (success && ib_downsampled) {
        if (!CreateAndAddFilamentEntity(downsampled_name, *buffer_builder, aabb,
                                        vb, ib_downsampled, material,
                                        BufferReuse::kYes)) {
            utility::LogWarning(
                    "Internal error: could not create downsampled point cloud");
        }
    }
    return success;
}

bool FilamentScene::AddGeometry(const std::string& object_name,
                                const t::geometry::PointCloud& point_cloud,
                                const Material& material,
                                const std::string& downsampled_name /*= ""*/,
                                size_t downsample_threshold /*= SIZE_MAX*/) {
    // Tensor::Min() and Tensor::Max() can be very slow on certain setups,
    // in particular macOS with clang 11.0.0. This is a temporary fix.
    auto ComputeAABB =
            [](const t::geometry::PointCloud& cloud) -> filament::Box {
        Eigen::Vector3f min_pt = {1e30f, 1e30f, 1e30f};
        Eigen::Vector3f max_pt = {-1e30f, -1e30f, -1e30f};
        const auto& points = cloud.GetPoints();
        const size_t n = points.GetSize();
        float* pts = (float*)points.AsTensor().GetDataPtr();
        for (size_t i = 0; i < 3 * n; i += 3) {
            min_pt[0] = std::min(min_pt[0], pts[i]);
            min_pt[1] = std::min(min_pt[1], pts[i + 1]);
            min_pt[2] = std::min(min_pt[2], pts[i + 2]);
            max_pt[0] = std::max(max_pt[0], pts[i]);
            max_pt[1] = std::max(max_pt[1], pts[i + 1]);
            max_pt[2] = std::max(max_pt[2], pts[i + 2]);
        }

        const filament::math::float3 min(min_pt.x(), min_pt.y(), min_pt.z());
        const filament::math::float3 max(max_pt.x(), max_pt.y(), max_pt.z());

        filament::Box aabb;
        aabb.set(min, max);
        return aabb;
    };

    // Basic sanity checks
    if (point_cloud.IsEmpty()) {
        utility::LogWarning("Point cloud for object {} is empty", object_name);
        return false;
    }
    const auto& points = point_cloud.GetPoints();
    if (points.GetDevice().GetType() == core::Device::DeviceType::CUDA) {
        utility::LogWarning(
                "GPU resident tensor point clouds are not supported at this "
                "time");
        return false;
    }
    if (points.GetDtype() != core::Dtype::Float32) {
        utility::LogWarning("tensor point cloud must have Dtype of Float32");
        return false;
    }

    auto buffer_builder = GeometryBuffersBuilder::GetBuilder(point_cloud);
    if (!downsampled_name.empty()) {
        buffer_builder->SetDownsampleThreshold(downsample_threshold);
    }
    auto buffers = buffer_builder->ConstructBuffers();
    auto vb = std::get<0>(buffers);
    auto ib = std::get<1>(buffers);
    auto ib_downsampled = std::get<2>(buffers);
    filament::Box aabb = ComputeAABB(point_cloud);
    bool success = CreateAndAddFilamentEntity(object_name, *buffer_builder,
                                              aabb, vb, ib, material);
    if (success && ib_downsampled) {
        if (!CreateAndAddFilamentEntity(downsampled_name, *buffer_builder, aabb,
                                        vb, ib_downsampled, material,
                                        BufferReuse::kYes)) {
            // If we failed to create a downsampled cloud, which would be
            // unlikely, create another entity with the original buffers
            // (since that succeeded).
            utility::LogWarning(
                    "Internal error: could not create downsampled point cloud");
            CreateAndAddFilamentEntity(downsampled_name, *buffer_builder, aabb,
                                       vb, ib, material, BufferReuse::kYes);
        }
    }
    return success;
}

#ifndef NDEBUG
void OutputMaterialProperties(const visualization::rendering::Material& mat) {
    utility::LogInfo("Material {}", mat.name);
    utility::LogInfo("\tAlpha: {}", mat.has_alpha);
    utility::LogInfo("\tBase Color: {},{},{},{}", mat.base_color.x(),
                     mat.base_color.y(), mat.base_color.z(),
                     mat.base_color.w());
    utility::LogInfo("\tBase Metallic: {}", mat.base_metallic);
    utility::LogInfo("\tBase Roughness: {}", mat.base_roughness);
    utility::LogInfo("\tBase Reflectance: {}", mat.base_reflectance);
    utility::LogInfo("\tBase Clear Cout: {}", mat.base_clearcoat);
}
#endif

bool FilamentScene::AddGeometry(const std::string& object_name,
                                const TriangleMeshModel& model) {
    if (geometries_.count(object_name) > 0 ||
        model_geometries_.count(object_name) > 0) {
        utility::LogWarning("Model {} has already been added to scene graph.",
                            object_name);
        return false;
    }

    std::vector<std::string> mesh_object_names;
    std::unordered_multiset<std::string> check_duplicates;
    for (const auto& mesh : model.meshes_) {
        auto& mat = model.materials_[mesh.material_idx];
        std::string derived_name(object_name + ":" + mesh.mesh_name);
        check_duplicates.insert(derived_name);
        if (check_duplicates.count(derived_name) > 1) {
            derived_name +=
                    std::string("_") +
                    std::to_string(check_duplicates.count(derived_name));
        }
        AddGeometry(derived_name, *(mesh.mesh), mat);
        mesh_object_names.push_back(derived_name);
    }
    model_geometries_[object_name] = mesh_object_names;

    return true;
}

bool FilamentScene::CreateAndAddFilamentEntity(
        const std::string& object_name,
        GeometryBuffersBuilder& buffer_builder,
        filament::Box& aabb,
        VertexBufferHandle vb,
        IndexBufferHandle ib,
        const Material& material,
        BufferReuse reusing_vertex_buffer /*= kNo*/) {
    auto vbuf = resource_mgr_.GetVertexBuffer(vb).lock();
    auto ibuf = resource_mgr_.GetIndexBuffer(ib).lock();

    auto filament_entity = utils::EntityManager::get().create();
    filament::RenderableManager::Builder builder(1);
    builder.boundingBox(aabb)
            .layerMask(FilamentView::kAllLayersMask, FilamentView::kMainLayer)
            .castShadows(true)
            .receiveShadows(true)
            .geometry(0, buffer_builder.GetPrimitiveType(), vbuf.get(),
                      ibuf.get());

    auto material_instance =
            AssignMaterialToFilamentGeometry(builder, material);

    auto result = builder.build(engine_, filament_entity);
    if (result == filament::RenderableManager::Builder::Success) {
        scene_->addEntity(filament_entity);

        auto giter = geometries_.emplace(std::make_pair(
                object_name,
                RenderableGeometry{object_name,
                                   true,
                                   true,
                                   true,
                                   {{}, material, material_instance},
                                   filament_entity,
                                   vb,
                                   ib}));

        SetGeometryTransform(object_name, Transform::Identity());
        UpdateMaterialProperties(giter.first->second);
    } else {
        // NOTE: Is there a better way to handle builder failing? That's a
        // sign of a major problem.
        utility::LogWarning(
                "Failed to build Filament resources for geometry {}",
                object_name);
        return false;
    }

    if (reusing_vertex_buffer == BufferReuse::kYes) {
        resource_mgr_.ReuseVertexBuffer(vb);
    }

    return true;
}

bool FilamentScene::HasGeometry(const std::string& object_name) const {
    if (GeometryIsModel(object_name)) {
        return true;
    }
    auto geom_entry = geometries_.find(object_name);
    return (geom_entry != geometries_.end());
}

static void deallocate_vertex_buffer(void* buffer,
                                     size_t size,
                                     void* user_ptr) {
    free(buffer);
}

void FilamentScene::UpdateGeometry(const std::string& object_name,
                                   const t::geometry::PointCloud& point_cloud,
                                   uint32_t update_flags) {
    auto geoms = GetGeometry(object_name, false);
    if (!geoms.empty()) {
        // Note: There should only be a single entry in geoms
        auto* g = geoms[0];
        auto vbuf_ptr = resource_mgr_.GetVertexBuffer(g->vb).lock();
        auto vbuf = vbuf_ptr.get();

        const auto& points = point_cloud.GetPoints();
        const size_t n_vertices = points.GetSize();

        // NOTE: number of points in the updated point cloud must be the
        // same as the number of points when the vertex buffer was first
        // created. If the number of points has changed then it cannot be
        // updated. In that case, you must remove the geometry then add it
        // again.
        if (n_vertices != vbuf->getVertexCount()) {
            utility::LogWarning(
                    "Geometry for point cloud {} cannot be updated because the "
                    "number of points has changed (Old: {}, New: {})",
                    object_name, vbuf->getVertexCount(), n_vertices);
            return;
        }

        if (update_flags & kUpdatePointsFlag) {
            filament::VertexBuffer::BufferDescriptor pts_descriptor(
                    points.AsTensor().GetDataPtr(),
                    n_vertices * 3 * sizeof(float));
            vbuf->setBufferAt(engine_, 0, std::move(pts_descriptor));
        }

        if (update_flags & kUpdateColorsFlag && point_cloud.HasPointColors()) {
            const size_t color_array_size = n_vertices * 3 * sizeof(float);
            filament::VertexBuffer::BufferDescriptor color_descriptor(
                    point_cloud.GetPointColors().AsTensor().GetDataPtr(),
                    color_array_size);
            vbuf->setBufferAt(engine_, 1, std::move(color_descriptor));
        }

        if (update_flags & kUpdateNormalsFlag &&
            point_cloud.HasPointNormals()) {
            const size_t normal_array_size = n_vertices * 4 * sizeof(float);
            const auto& normals = point_cloud.GetPointNormals();

            // Converting normals to Filament type - quaternions
            auto float4v_tangents = static_cast<filament::math::quatf*>(
                    malloc(normal_array_size));
            auto orientation =
                    filament::geometry::SurfaceOrientation::Builder()
                            .vertexCount(n_vertices)
                            .normals(reinterpret_cast<filament::math::float3*>(
                                    normals.AsTensor().GetDataPtr()))
                            .build();
            orientation->getQuats(float4v_tangents, n_vertices);
            filament::VertexBuffer::BufferDescriptor normals_descriptor(
                    float4v_tangents, normal_array_size,
                    deallocate_vertex_buffer);
            vbuf->setBufferAt(engine_, 2, std::move(normals_descriptor));
        }

        if (update_flags & kUpdateUv0Flag) {
            const size_t uv_array_size = n_vertices * 2 * sizeof(float);
            if (point_cloud.HasPointAttr("uv")) {
                filament::VertexBuffer::BufferDescriptor uv_descriptor(
                        point_cloud.GetPointAttr("uv").AsTensor().GetDataPtr(),
                        uv_array_size);
                vbuf->setBufferAt(engine_, 3, std::move(uv_descriptor));
            } else if (point_cloud.HasPointAttr("__visualization_scalar")) {
                // Update in PointCloudBuffers.cpp, too:
                //     TPointCloudBuffersBuilder::ConstructBuffers
                float* uv_array = static_cast<float*>(malloc(uv_array_size));
                memset(uv_array, 0, uv_array_size);
                float* src = static_cast<float*>(
                        point_cloud.GetPointAttr("__visualization_scalar")
                                .AsTensor()
                                .GetDataPtr());
                const size_t n = 2 * n_vertices;
                for (size_t i = 0; i < n; i += 2) {
                    uv_array[i] = *src++;
                }
                filament::VertexBuffer::BufferDescriptor uv_descriptor(
                        uv_array, uv_array_size, DeallocateBuffer);
                vbuf->setBufferAt(engine_, 3, std::move(uv_descriptor));
            }
        }
    }
}

void FilamentScene::RemoveGeometry(const std::string& object_name) {
    auto geoms = GetGeometry(object_name, false);
    if (!geoms.empty()) {
        for (auto* g : geoms) {
            scene_->remove(g->filament_entity);
            g->ReleaseResources(engine_, resource_mgr_);
            geometries_.erase(g->name);
        }
    }

    if (GeometryIsModel(object_name)) {
        model_geometries_.erase(object_name);
    }
}

void FilamentScene::ShowGeometry(const std::string& object_name, bool show) {
    auto geoms = GetGeometry(object_name);
    for (auto* g : geoms) {
        if (g->visible != show) {
            g->visible = show;
            if (show) {
                scene_->addEntity(g->filament_entity);
            } else {
                scene_->remove(g->filament_entity);
            }
        }
    }
}

bool FilamentScene::GeometryIsVisible(const std::string& object_name) {
    auto geoms = GetGeometry(object_name);
    if (!geoms.empty()) {
        // NOTE: all meshes of model share same visibility so we only need to
        // check first entry of this array
        return geoms[0]->visible;
    } else {
        return false;
    }
}

utils::EntityInstance<filament::TransformManager>
FilamentScene::GetGeometryTransformInstance(RenderableGeometry* geom) {
    filament::TransformManager::Instance itransform;
    auto& transform_mgr = engine_.getTransformManager();
    itransform = transform_mgr.getInstance(geom->filament_entity);
    if (!itransform.isValid()) {
        using namespace filament::math;
        transform_mgr.create(geom->filament_entity);
        itransform = transform_mgr.getInstance(geom->filament_entity);
        transform_mgr.create(geom->filament_entity, itransform,
                             mat4f::translation(float3{0.0f, 0.0f, 0.0f}));
    }
    return itransform;
}

void FilamentScene::SetGeometryTransform(const std::string& object_name,
                                         const Transform& transform) {
    auto geoms = GetGeometry(object_name);
    for (auto* g : geoms) {
        auto itransform = GetGeometryTransformInstance(g);
        if (itransform.isValid()) {
            const auto& ematrix = transform.matrix();
            auto& transform_mgr = engine_.getTransformManager();
            transform_mgr.setTransform(
                    itransform,
                    converters::FilamentMatrixFromEigenMatrix(ematrix));
        }
    }
}

FilamentScene::Transform FilamentScene::GetGeometryTransform(
        const std::string& object_name) {
    Transform etransform;
    auto geoms = GetGeometry(object_name);
    if (!geoms.empty()) {
        auto itransform = GetGeometryTransformInstance(geoms[0]);
        if (itransform.isValid()) {
            auto& transform_mgr = engine_.getTransformManager();
            auto ftransform = transform_mgr.getTransform(itransform);
            etransform = converters::EigenMatrixFromFilamentMatrix(ftransform);
        }
    }
    return etransform;
}

geometry::AxisAlignedBoundingBox FilamentScene::GetGeometryBoundingBox(
        const std::string& object_name) {
    geometry::AxisAlignedBoundingBox result;
    auto geoms = GetGeometry(object_name);
    for (auto* g : geoms) {
        auto& renderable_mgr = engine_.getRenderableManager();
        auto inst = renderable_mgr.getInstance(g->filament_entity);
        auto box = renderable_mgr.getAxisAlignedBoundingBox(inst);

        auto& transform_mgr = engine_.getTransformManager();
        auto itransform = transform_mgr.getInstance(g->filament_entity);
        auto transform = transform_mgr.getWorldTransform(itransform);

        box = rigidTransform(box, transform);

        auto min = box.center - box.halfExtent;
        auto max = box.center + box.halfExtent;
        result += {{min.x, min.y, min.z}, {max.x, max.y, max.z}};
    }
    return result;
}

void FilamentScene::GeometryShadows(const std::string& object_name,
                                    bool cast_shadows,
                                    bool receive_shadows) {
    auto geoms = GetGeometry(object_name);
    for (auto* g : geoms) {
        auto& renderable_mgr = engine_.getRenderableManager();
        filament::RenderableManager::Instance inst =
                renderable_mgr.getInstance(g->filament_entity);
        renderable_mgr.setCastShadows(inst, cast_shadows);
        renderable_mgr.setReceiveShadows(inst, receive_shadows);
    }
}

void FilamentScene::UpdateDefaultLit(GeometryMaterialInstance& geom_mi) {
    auto& material = geom_mi.properties;
    auto& maps = geom_mi.maps;

    renderer_.ModifyMaterial(geom_mi.mat_instance)
            .SetColor("baseColor", material.base_color, false)
            .SetParameter("pointSize", material.point_size)
            .SetParameter("baseRoughness", material.base_roughness)
            .SetParameter("baseMetallic", material.base_metallic)
            .SetParameter("reflectance", material.base_reflectance)
            .SetParameter("clearCoat", material.base_clearcoat)
            .SetParameter("clearCoatRoughness",
                          material.base_clearcoat_roughness)
            .SetParameter("anisotropy", material.base_anisotropy)
            .SetTexture("albedo", maps.albedo_map,
                        rendering::TextureSamplerParameters::Pretty())
            .SetTexture("normalMap", maps.normal_map,
                        rendering::TextureSamplerParameters::Pretty())
            .SetTexture("ao_rough_metalMap", maps.ao_rough_metal_map,
                        rendering::TextureSamplerParameters::Pretty())
            .SetTexture("reflectanceMap", maps.reflectance_map,
                        rendering::TextureSamplerParameters::Pretty())
            // NOTE: Disabled temporarily to avoid Filament warning until
            // defaultLit is reworked to use fewer samplers
            // .SetTexture("clearCoatMap", maps.clear_coat_map,
            //             rendering::TextureSamplerParameters::Pretty())
            // .SetTexture("clearCoatRoughnessMap",
            // maps.clear_coat_roughness_map,
            //             rendering::TextureSamplerParameters::Pretty())
            .SetTexture("anisotropyMap", maps.anisotropy_map,
                        rendering::TextureSamplerParameters::Pretty())
            .Finish();
}

void FilamentScene::UpdateDefaultUnlit(GeometryMaterialInstance& geom_mi) {
    renderer_.ModifyMaterial(geom_mi.mat_instance)
            .SetColor("baseColor", geom_mi.properties.base_color, true)
            .SetParameter("pointSize", geom_mi.properties.point_size)
            .SetTexture("albedo", geom_mi.maps.albedo_map,
                        rendering::TextureSamplerParameters::Pretty())
            .Finish();
}

void FilamentScene::UpdateNormalShader(GeometryMaterialInstance& geom_mi) {
    renderer_.ModifyMaterial(geom_mi.mat_instance)
            .SetParameter("pointSize", geom_mi.properties.point_size)
            .Finish();
}

void FilamentScene::UpdateDepthShader(GeometryMaterialInstance& geom_mi) {
    auto* camera = views_.begin()->second.view->GetCamera();
    const float f = float(camera->GetFar());
    const float n = float(camera->GetNear());
    renderer_.ModifyMaterial(geom_mi.mat_instance)
            .SetParameter("pointSize", geom_mi.properties.point_size)
            .SetParameter("cameraNear", n)
            .SetParameter("cameraFar", f)
            .Finish();
}

void FilamentScene::UpdateGradientShader(GeometryMaterialInstance& geom_mi) {
    bool isLUT =
            (geom_mi.properties.gradient->GetMode() == Gradient::Mode::kLUT);
    renderer_.ModifyMaterial(geom_mi.mat_instance)
            .SetParameter("minValue", geom_mi.properties.scalar_min)
            .SetParameter("maxValue", geom_mi.properties.scalar_max)
            .SetParameter("isLUT", (isLUT ? 1.0f : 0.0f))
            .SetParameter("pointSize", geom_mi.properties.point_size)
            .SetTexture(
                    "gradient", geom_mi.maps.gradient_texture,
                    isLUT ? rendering::TextureSamplerParameters::Simple()
                          : rendering::TextureSamplerParameters::LinearClamp())
            .Finish();
}

void FilamentScene::UpdateSolidColorShader(GeometryMaterialInstance& geom_mi) {
    renderer_.ModifyMaterial(geom_mi.mat_instance)
            .SetColor("baseColor", geom_mi.properties.base_color, true)
            .SetParameter("pointSize", geom_mi.properties.point_size)
            .Finish();
}

std::shared_ptr<geometry::Image> CombineTextures(
        std::shared_ptr<geometry::Image> ao,
        std::shared_ptr<geometry::Image> rough,
        std::shared_ptr<geometry::Image> metal) {
    int width = 0, height = 0;
    if (ao && ao->HasData()) {
        width = ao->width_;
        height = ao->height_;
    }
    if (rough && rough->HasData()) {
        if (width == 0) {
            width = rough->width_;
            height = rough->height_;
        } else if (width != rough->width_ || height != rough->height_) {
            utility::LogWarning(
                    "Attribute texture maps must have same dimensions");
            return {};
        }
    }
    if (metal && metal->HasData()) {
        if (width == 0) {
            width = metal->width_;
            height = metal->height_;
        } else if (width != metal->width_ || height != metal->height_) {
            utility::LogWarning(
                    "Attribute texture maps must have same dimensions");
            return {};
        }
    }

    // no maps are valid so return empty texture and let caller use defaults
    if (width == 0 || height == 0) {
        return {};
    }

    auto image = std::make_shared<geometry::Image>();
    image->Prepare(width, height, 3, 1);
    auto data = reinterpret_cast<uint8_t*>(image->data_.data());

    auto set_pixel = [&data](std::shared_ptr<geometry::Image> map, int i,
                             int j) {
        if (map && map->HasData()) {
            *data++ = *(map->PointerAt<uint8_t>(j, i, 0));
        } else {
            *data++ = 255;
        }
    };

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            set_pixel(ao, i, j);
            set_pixel(rough, i, j);
            set_pixel(metal, i, j);
        }
    }

    return image;
}

void CombineTextures(std::shared_ptr<geometry::Image> ao,
                     std::shared_ptr<geometry::Image> rough_metal) {
    int width = rough_metal->width_;
    int height = rough_metal->height_;

    if (ao && ao->HasData()) {
        if (width != ao->width_ || height != ao->height_) {
            utility::LogWarning(
                    "Attribute texture maps must have same dimensions");
            return;
        }
    }

    auto data = reinterpret_cast<uint8_t*>(rough_metal->data_.data());

    auto stride = rough_metal->num_of_channels_;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            if (ao && ao->HasData()) {
                *data = *(ao->PointerAt<uint8_t>(j, i, 0));
            } else {
                *data = 255;
            }
            data += stride;
        }
    }
}

void FilamentScene::UpdateMaterialProperties(RenderableGeometry& geom) {
    auto& props = geom.mat.properties;
    auto& maps = geom.mat.maps;

    // Load textures
    auto is_map_valid = [](std::shared_ptr<geometry::Image> map) -> bool {
        return map && map->HasData();
    };
    if (is_map_valid(props.albedo_img)) {
        maps.albedo_map = renderer_.AddTexture(props.albedo_img, true);
    }
    if (is_map_valid(props.normal_img)) {
        maps.normal_map = renderer_.AddTexture(props.normal_img);
    }
    if (is_map_valid(props.reflectance_img)) {
        maps.reflectance_map = renderer_.AddTexture(props.reflectance_img);
    }
    if (is_map_valid(props.clearcoat_img)) {
        maps.clear_coat_map = renderer_.AddTexture(props.clearcoat_img);
    }
    if (is_map_valid(props.clearcoat_roughness_img)) {
        maps.clear_coat_roughness_map =
                renderer_.AddTexture(props.clearcoat_roughness_img);
    }
    if (is_map_valid(props.anisotropy_img)) {
        maps.anisotropy_map = renderer_.AddTexture(props.anisotropy_img);
    }
    if (props.shader == "unlitGradient") {
        maps.gradient_texture = props.gradient->GetTextureHandle(renderer_);
    }

    // Create combined ao/rough/metal texture
    if (is_map_valid(props.ao_rough_metal_img)) {
        CombineTextures(props.ao_img, props.ao_rough_metal_img);
        maps.ao_rough_metal_map =
                renderer_.AddTexture(props.ao_rough_metal_img);
    } else if (is_map_valid(props.ao_img) ||
               is_map_valid(props.roughness_img) ||
               is_map_valid(props.metallic_img)) {
        props.ao_rough_metal_img = CombineTextures(
                props.ao_img, props.roughness_img, props.metallic_img);
        maps.ao_rough_metal_map =
                renderer_.AddTexture(props.ao_rough_metal_img);
    }

    // Update shader properties
    // TODO: Use a functional interface to get appropriate update methods
    if (props.shader == "defaultLit" ||
        props.shader == "defaultLitTransparency") {
        UpdateDefaultLit(geom.mat);
    } else if (props.shader == "defaultUnlit") {
        UpdateDefaultUnlit(geom.mat);
    } else if (props.shader == "normals") {
        UpdateNormalShader(geom.mat);
    } else if (props.shader == "depth") {
        UpdateDepthShader(geom.mat);
    } else if (props.shader == "unlitGradient") {
        UpdateGradientShader(geom.mat);
    } else if (props.shader == "unlitSolidColor") {
        UpdateSolidColorShader(geom.mat);
    }
}

void FilamentScene::OverrideMaterialInternal(RenderableGeometry* geom,
                                             const Material& material,
                                             bool shader_only) {
    // Has the shader changed?
    if (geom->mat.properties.shader != material.shader) {
        // TODO: put this in a method
        auto shader = defaults_mapping::shader_mappings[material.shader];
        if (!shader) shader = defaults_mapping::kColorOnlyMesh;
        auto old_mi = geom->mat.mat_instance;
        auto new_mi = resource_mgr_.CreateMaterialInstance(shader);
        auto wmat_instance = resource_mgr_.GetMaterialInstance(new_mi);
        if (!wmat_instance.expired()) {
            auto& renderable_mgr = engine_.getRenderableManager();
            filament::RenderableManager::Instance inst =
                    renderable_mgr.getInstance(geom->filament_entity);
            renderable_mgr.setMaterialInstanceAt(inst, 0,
                                                 wmat_instance.lock().get());
        }
        geom->mat.mat_instance = new_mi;
        resource_mgr_.Destroy(old_mi);
    }
    geom->mat.properties = material;
    if (shader_only) {
        if (material.shader == "defaultLit" ||
            material.shader == "defaultLitTransparency") {
            UpdateDefaultLit(geom->mat);
        } else if (material.shader == "defaultUnlit") {
            UpdateDefaultUnlit(geom->mat);
        } else if (material.shader == "normals") {
            UpdateNormalShader(geom->mat);
        } else if (material.shader == "unlitGradient") {
            UpdateGradientShader(geom->mat);
        } else if (material.shader == "unlitColorMap") {
            UpdateGradientShader(geom->mat);
        } else if (material.shader == "unlitSolidColor") {
            UpdateSolidColorShader(geom->mat);
        } else {
            UpdateDepthShader(geom->mat);
        }
    } else {
        UpdateMaterialProperties(*geom);
    }
}

void FilamentScene::OverrideMaterial(const std::string& object_name,
                                     const Material& material) {
    auto geoms = GetGeometry(object_name);
    for (auto* g : geoms) {
        OverrideMaterialInternal(g, material);
    }
}

void FilamentScene::QueryGeometry(std::vector<std::string>& geometry) {
    for (const auto& ge : geometries_) {
        geometry.push_back(ge.first);
    }
}

void FilamentScene::OverrideMaterialAll(const Material& material,
                                        bool shader_only) {
    for (auto& ge : geometries_) {
        OverrideMaterialInternal(&ge.second, material, shader_only);
    }
}

bool FilamentScene::AddPointLight(const std::string& light_name,
                                  const Eigen::Vector3f& color,
                                  const Eigen::Vector3f& position,
                                  float intensity,
                                  float falloff,
                                  bool cast_shadows) {
    if (lights_.count(light_name) > 0) {
        utility::LogWarning(
                "Cannot add point light because {} has already been added",
                light_name);
        return false;
    }

    filament::LightManager::Type light_type =
            filament::LightManager::Type::POINT;
    auto light = utils::EntityManager::get().create();
    auto result = filament::LightManager::Builder(light_type)
                          .position({position.x(), position.y(), position.z()})
                          .intensity(intensity)
                          .falloff(falloff)
                          .castShadows(cast_shadows)
                          .color({color.x(), color.y(), color.z()})
                          .build(engine_, light);

    if (result == filament::LightManager::Builder::Success) {
        lights_.emplace(std::make_pair(light_name, LightEntity{true, light}));
    } else {
        utility::LogWarning("Failed to build Filament light resources for {}",
                            light_name);
        return false;
    }

    return true;
}

bool FilamentScene::AddSpotLight(const std::string& light_name,
                                 const Eigen::Vector3f& color,
                                 const Eigen::Vector3f& position,
                                 const Eigen::Vector3f& direction,
                                 float intensity,
                                 float falloff,
                                 float inner_cone_angle,
                                 float outer_cone_angle,
                                 bool cast_shadows) {
    if (lights_.count(light_name) > 0) {
        utility::LogWarning(
                "Cannot add point light because {} has already been added",
                light_name);
        return false;
    }

    filament::LightManager::Type light_type =
            filament::LightManager::Type::SPOT;
    auto light = utils::EntityManager::get().create();
    auto result =
            filament::LightManager::Builder(light_type)
                    .direction({direction.x(), direction.y(), direction.z()})
                    .position({position.x(), position.y(), position.z()})
                    .intensity(intensity)
                    .falloff(falloff)
                    .castShadows(cast_shadows)
                    .color({color.x(), color.y(), color.z()})
                    .spotLightCone(inner_cone_angle, outer_cone_angle)
                    .build(engine_, light);

    if (result == filament::LightManager::Builder::Success) {
        lights_.emplace(std::make_pair(light_name, LightEntity{true, light}));
    } else {
        utility::LogWarning("Failed to build Filament light resources for {}",
                            light_name);
        return false;
    }

    return true;
}

Light& FilamentScene::GetLight(const std::string& light_name) {
    // TODO: Not yet implemented. I still don't see any advantage to doing this
    static Light blah;
    return blah;
}

void FilamentScene::UpdateLight(const std::string& light_name,
                                const Light& light) {
    // TODO: Not yet implemented. I still don't see any advantage to doing this
}

void FilamentScene::RemoveLight(const std::string& light_name) {
    auto light = GetLightInternal(light_name);
    if (light) {
        scene_->remove(light->filament_entity);
        engine_.destroy(light->filament_entity);
        lights_.erase(light_name);
    }
}

void FilamentScene::UpdateLightColor(const std::string& light_name,
                                     const Eigen::Vector3f& color) {
    auto light = GetLightInternal(light_name);
    if (light) {
        auto& light_mgr = engine_.getLightManager();
        filament::LightManager::Instance inst =
                light_mgr.getInstance(light->filament_entity);
        light_mgr.setColor(inst, {color(0), color(1), color(2)});
    }
}

void FilamentScene::UpdateLightPosition(const std::string& light_name,
                                        const Eigen::Vector3f& position) {
    auto light = GetLightInternal(light_name);
    if (light) {
        auto& light_mgr = engine_.getLightManager();
        filament::LightManager::Instance inst =
                light_mgr.getInstance(light->filament_entity);
        if (!light_mgr.isDirectional(inst)) {
            light_mgr.setPosition(inst,
                                  {position.x(), position.y(), position.z()});
        }
    }
}

void FilamentScene::UpdateLightDirection(const std::string& light_name,
                                         const Eigen::Vector3f& direction) {
    auto light = GetLightInternal(light_name);
    if (light) {
        auto& light_mgr = engine_.getLightManager();
        filament::LightManager::Instance inst =
                light_mgr.getInstance(light->filament_entity);
        light_mgr.setDirection(inst,
                               {direction.x(), direction.y(), direction.z()});
    }
}

void FilamentScene::UpdateLightIntensity(const std::string& light_name,
                                         float intensity) {
    auto light = GetLightInternal(light_name);
    if (light) {
        auto& light_mgr = engine_.getLightManager();
        filament::LightManager::Instance inst =
                light_mgr.getInstance(light->filament_entity);
        light_mgr.setIntensity(inst, intensity);
    }
}

void FilamentScene::UpdateLightFalloff(const std::string& light_name,
                                       float falloff) {
    auto light = GetLightInternal(light_name);
    if (light) {
        auto& light_mgr = engine_.getLightManager();
        filament::LightManager::Instance inst =
                light_mgr.getInstance(light->filament_entity);
        light_mgr.setFalloff(inst, falloff);
    }
}

void FilamentScene::UpdateLightConeAngles(const std::string& light_name,
                                          float inner_cone_angle,
                                          float outer_cone_angle) {
    // TODO: Research. Not previously implemented
}

void FilamentScene::EnableLightShadow(const std::string& light_name,
                                      bool cast_shadows) {
    // TODO: Research. Not previously implemented.
}

void FilamentScene::CreateSunDirectionalLight() {
    filament::LightManager::Type light_type = filament::LightManager::Type::SUN;
    auto light = utils::EntityManager::get().create();
    auto result = filament::LightManager::Builder(light_type)
                          .direction({0.f, 0.f, 1.f})
                          .intensity(10000.0f)
                          .falloff(10.0)
                          .castShadows(true)
                          .color({1.f, 1.f, 1.f})
                          .build(engine_, light);

    if (result == filament::LightManager::Builder::Success) {
        sun_.filament_entity = light;
        scene_->addEntity(sun_.filament_entity);
    } else {
        utility::LogWarning(
                "Failed to build Filament light resources for sun light");
    }
}

void FilamentScene::SetDirectionalLight(const Eigen::Vector3f& direction,
                                        const Eigen::Vector3f& color,
                                        float intensity) {
    auto& light_mgr = engine_.getLightManager();
    filament::LightManager::Instance inst =
            light_mgr.getInstance(sun_.filament_entity);
    light_mgr.setDirection(inst, {direction.x(), direction.y(), direction.z()});
    light_mgr.setColor(inst, {color.x(), color.y(), color.z()});
    light_mgr.setIntensity(inst, intensity);
}

void FilamentScene::EnableDirectionalLight(bool enable) {
    if (sun_.enabled != enable) {
        sun_.enabled = enable;
        if (enable) {
            scene_->addEntity(sun_.filament_entity);
        } else {
            scene_->remove(sun_.filament_entity);
        }
    }
}

void FilamentScene::EnableDirectionalLightShadows(bool enable) {
    // TODO: Research. Not previously implemented
}

void FilamentScene::SetDirectionalLightDirection(
        const Eigen::Vector3f& direction) {
    auto& light_mgr = engine_.getLightManager();
    filament::LightManager::Instance inst =
            light_mgr.getInstance(sun_.filament_entity);
    light_mgr.setDirection(inst, {direction.x(), direction.y(), direction.z()});
}

Eigen::Vector3f FilamentScene::GetDirectionalLightDirection() {
    auto& light_mgr = engine_.getLightManager();
    filament::LightManager::Instance inst =
            light_mgr.getInstance(sun_.filament_entity);
    auto dir = light_mgr.getDirection(inst);
    return {dir[0], dir[1], dir[2]};
}

bool FilamentScene::SetIndirectLight(const std::string& ibl_name) {
    // Load IBL
    std::string ibl_path = ibl_name + std::string("_ibl.ktx");
    rendering::IndirectLightHandle new_ibl =
            renderer_.AddIndirectLight(ResourceLoadRequest(ibl_path.c_str()));
    if (!new_ibl) {
        return false;
    }

    auto wlight = resource_mgr_.GetIndirectLight(new_ibl);
    if (auto light = wlight.lock()) {
        indirect_light_ = wlight;
        if (ibl_enabled_) scene_->setIndirectLight(light.get());
        ibl_name_ = ibl_name;
    }

    // Load matching skybox
    std::string skybox_path = ibl_name + std::string("_skybox.ktx");
    SkyboxHandle sky =
            renderer_.AddSkybox(ResourceLoadRequest(skybox_path.c_str()));
    auto wskybox = resource_mgr_.GetSkybox(sky);
    if (auto skybox = wskybox.lock()) {
        skybox_ = wskybox;
        if (skybox_enabled_) scene_->setSkybox(skybox.get());
    }

    return true;
}

const std::string& FilamentScene::GetIndirectLight() { return ibl_name_; }

void FilamentScene::EnableIndirectLight(bool enable) {
    if (enable != ibl_enabled_) {
        if (enable) {
            if (auto light = indirect_light_.lock()) {
                scene_->setIndirectLight(light.get());
            }
        } else {
            scene_->setIndirectLight(nullptr);
        }
        ibl_enabled_ = enable;
    }
}

void FilamentScene::SetIndirectLightIntensity(float intensity) {
    if (auto light = indirect_light_.lock()) {
        light->setIntensity(intensity);
    }
}

float FilamentScene::GetIndirectLightIntensity() {
    if (auto light = indirect_light_.lock()) {
        return light->getIntensity();
    }
    return 0.f;
}

void FilamentScene::SetIndirectLightRotation(const Transform& rotation) {
    if (auto light = indirect_light_.lock()) {
        auto ft = converters::FilamentMatrixFromEigenMatrix(rotation.matrix());
        light->setRotation(ft.upperLeft());
    }
}

FilamentScene::Transform FilamentScene::GetIndirectLightRotation() {
    if (auto light = indirect_light_.lock()) {
        converters::FilamentMatrix ft(light->getRotation());
        auto et = converters::EigenMatrixFromFilamentMatrix(ft);

        return Transform(et);
    }
    return {};
}

void FilamentScene::ShowSkybox(bool show) {
    if (show != skybox_enabled_) {
        if (show) {
            if (auto skybox = skybox_.lock()) {
                scene_->setSkybox(skybox.get());
            }
        } else {
            scene_->setSkybox(nullptr);
        }
        skybox_enabled_ = show;
    }
}

struct RenderRequest {
    bool frame_done = false;
    std::shared_ptr<geometry::Image> image;
};

void ReadPixelsCallback(void* buffer, size_t buffer_size, void* user) {
    auto rr = static_cast<RenderRequest*>(user);
    rr->frame_done = true;

    if (buffer_size > 0) {
        rr->image->data_ = std::vector<uint8_t>((uint8_t*)buffer,
                                                (uint8_t*)buffer + buffer_size);
    } else {
        utility::LogWarning(
                "0 buffer size encountered while rendering to image");
    }
}

void FilamentScene::RenderToImage(
        int width,
        int height,
        std::function<void(std::shared_ptr<geometry::Image>)> callback) {
    auto view = views_.begin()->second.view.get();
    renderer_.RenderToImage(width, height, view, this, callback);
}

std::vector<FilamentScene::RenderableGeometry*> FilamentScene::GetGeometry(
        const std::string& object_name, bool warn_if_not_found) {
    std::vector<RenderableGeometry*> geoms;
    if (GeometryIsModel(object_name)) {
        for (const auto& name : model_geometries_[object_name]) {
            auto geom_entry = geometries_.find(name);
            if (geom_entry == geometries_.end()) {
                if (warn_if_not_found) {
                    utility::LogWarning("Geometry {} is not in the scene graph",
                                        name);
                }
            } else {
                geoms.push_back(&geom_entry->second);
            }
        }
    } else {
        auto geom_entry = geometries_.find(object_name);
        if (geom_entry == geometries_.end()) {
            if (warn_if_not_found) {
                utility::LogWarning("Geometry {} is not in the scene graph",
                                    object_name);
            }
        } else {
            geoms.push_back(&geom_entry->second);
        }
    }

    return geoms;
}

bool FilamentScene::GeometryIsModel(const std::string& object_name) const {
    return model_geometries_.count(object_name) > 0;
}

FilamentScene::LightEntity* FilamentScene::GetLightInternal(
        const std::string& light_name, bool warn_if_not_found) {
    auto light_entry = lights_.find(light_name);
    if (light_entry == lights_.end()) {
        if (warn_if_not_found) {
            utility::LogWarning("Light {} is not in the scene graph",
                                light_name);
        }
        return nullptr;
    }
    return &(light_entry->second);
}

void FilamentScene::RenderableGeometry::ReleaseResources(
        filament::Engine& engine, FilamentResourceManager& manager) {
    if (vb) manager.Destroy(vb);
    if (ib) manager.Destroy(ib);
    engine.destroy(filament_entity);

    // Delete texture maps...
    auto destroy_map = [&manager](rendering::TextureHandle map) {
        if (map && map != rendering::FilamentResourceManager::kDefaultTexture &&
            map != rendering::FilamentResourceManager::kDefaultNormalMap)
            manager.Destroy(map);
    };
    destroy_map(mat.maps.albedo_map);
    destroy_map(mat.maps.normal_map);
    destroy_map(mat.maps.ao_rough_metal_map);
    destroy_map(mat.maps.reflectance_map);
    destroy_map(mat.maps.clear_coat_map);
    destroy_map(mat.maps.clear_coat_roughness_map);
    destroy_map(mat.maps.anisotropy_map);

    manager.Destroy(mat.mat_instance);

    filament_entity.clear();
}

void FilamentScene::Draw(filament::Renderer& renderer) {
    for (auto& pair : views_) {
        auto& container = pair.second;
        // Skip inactive views
        if (!container.is_active) continue;
        if (container.render_count-- == 0) {
            container.is_active = false;
            continue;
        }

        container.view->PreRender();
        renderer.render(container.view->GetNativeView());
        container.view->PostRender();
    }
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
