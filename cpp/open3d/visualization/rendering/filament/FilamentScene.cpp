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
#include <filament/View.h>
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
#include "open3d/utility/Console.h"
#include "open3d/visualization/rendering/Light.h"
#include "open3d/visualization/rendering/Material.h"
#include "open3d/visualization/rendering/Model.h"
#include "open3d/visualization/rendering/RendererHandle.h"
#include "open3d/visualization/rendering/filament/FilamentEntitiesMods.h"
#include "open3d/visualization/rendering/filament/FilamentGeometryBuffersBuilder.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"
#include "open3d/visualization/rendering/filament/FilamentResourceManager.h"
#include "open3d/visualization/rendering/filament/FilamentView.h"

namespace {  // avoid polluting global namespace, since only used here
/// @cond

namespace defaults_mapping {

using GeometryType = open3d::geometry::Geometry::GeometryType;
using MaterialHandle = open3d::visualization::rendering::MaterialHandle;
using ResourceManager =
        open3d::visualization::rendering::FilamentResourceManager;

std::unordered_map<std::string, MaterialHandle> shader_mappings = {
        {"defaultLit", ResourceManager::kDefaultLit},
        {"defaultUnlit", ResourceManager::kDefaultUnlit},
        {"normals", ResourceManager::kDefaultNormalShader},
        {"depth", ResourceManager::kDefaultDepthShader}};

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
                                const Material& material) {
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

    auto buffers = geometry_buffer_builder->ConstructBuffers();
    auto vb = std::get<0>(buffers);
    auto ib = std::get<1>(buffers);

    filament::Box aabb = geometry_buffer_builder->ComputeAABB();

    auto vbuf = resource_mgr_.GetVertexBuffer(vb).lock();
    auto ibuf = resource_mgr_.GetIndexBuffer(ib).lock();

    auto filament_entity = utils::EntityManager::get().create();
    filament::RenderableManager::Builder builder(1);
    builder.boundingBox(aabb)
            .layerMask(FilamentView::kAllLayersMask, FilamentView::kMainLayer)
            .castShadows(true)
            .receiveShadows(true)
            .geometry(0, geometry_buffer_builder->GetPrimitiveType(),
                      vbuf.get(), ibuf.get());

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

    return true;
}

bool FilamentScene::AddGeometry(const std::string& object_name,
                                const TriangleMeshModel& model) {
    if (geometries_.count(object_name) > 0 ||
        model_geometries_.count(object_name) > 0) {
        utility::LogWarning("Model {} has already been added to scene graph.",
                            object_name);
        return false;
    }

    std::vector<std::string> mesh_object_names;
    for (const auto& mesh : model.meshes_) {
        auto& mat = model.materials_[mesh.material_idx];
        std::string derived_name(object_name + ":" + mesh.mesh_name);
        AddGeometry(derived_name, *(mesh.mesh), mat);
        mesh_object_names.push_back(derived_name);
    }
    model_geometries_[object_name] = mesh_object_names;

    return true;
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
            .SetColor("baseColor", material.base_color, true)
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

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            if (ao && ao->HasData()) {
                *data = *(ao->PointerAt<uint8_t>(j, i, 0));
            } else {
                *data = 255;
            }
            data += 3;
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
    if (props.shader == "defaultLit") {
        UpdateDefaultLit(geom.mat);
    } else if (props.shader == "defaultUnlit") {
        UpdateDefaultUnlit(geom.mat);
    } else if (props.shader == "normals") {
        UpdateNormalShader(geom.mat);
    } else if (props.shader == "depth") {
        UpdateDepthShader(geom.mat);
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
        if (material.shader == "defaultLit") {
            UpdateDefaultLit(geom->mat);
        } else if (material.shader == "defaultUnlit") {
            UpdateDefaultUnlit(geom->mat);
        } else if (material.shader == "normals") {
            UpdateNormalShader(geom->mat);
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

bool FilamentScene::GeometryIsModel(const std::string& object_name) {
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
    // NOTE: is this really necessary?
    filament_entity.clear();
}

void FilamentScene::Draw(filament::Renderer& renderer) {
    for (const auto& pair : views_) {
        auto& container = pair.second;
        if (container.is_active) {
            container.view->PreRender();
            renderer.render(container.view->GetNativeView());
            container.view->PostRender();
        }
    }
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
