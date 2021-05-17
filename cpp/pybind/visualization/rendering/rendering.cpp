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

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/visualization/rendering/ColorGrading.h"
#include "open3d/visualization/rendering/Gradient.h"
#include "open3d/visualization/rendering/Material.h"
#include "open3d/visualization/rendering/Model.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/Renderer.h"
#include "open3d/visualization/rendering/Scene.h"
#include "open3d/visualization/rendering/View.h"
#include "open3d/visualization/rendering/filament/FilamentEngine.h"
#include "open3d/visualization/rendering/filament/FilamentRenderer.h"
#include "pybind/docstring.h"
#include "pybind/visualization/gui/gui.h"
#include "pybind/visualization/visualization.h"
#include "pybind11/functional.h"

namespace open3d {
namespace visualization {
namespace rendering {

class PyOffscreenRenderer {
public:
    PyOffscreenRenderer(int width,
                        int height,
                        const std::string &resource_path,
                        bool headless) {
        gui::InitializeForPython(resource_path);
        width_ = width;
        height_ = height;
        if (headless) {
            EngineInstance::EnableHeadless();
        }
        renderer_ = new FilamentRenderer(EngineInstance::GetInstance(), width,
                                         height,
                                         EngineInstance::GetResourceManager());
        scene_ = new Open3DScene(*renderer_);
    }

    ~PyOffscreenRenderer() {
        delete scene_;
        delete renderer_;
    }

    Open3DScene *GetScene() { return scene_; }

    std::shared_ptr<geometry::Image> RenderToImage() {
        return gui::RenderToImageWithoutWindow(scene_, width_, height_);
    }

    std::shared_ptr<geometry::Image> RenderToDepthImage() {
        return gui::RenderToDepthImageWithoutWindow(scene_, width_, height_);
    }

    void SetupCamera(const camera::PinholeCameraIntrinsic &intrinsic,
                     const Eigen::Matrix4d &extrinsic) {
        SetupCamera(intrinsic.intrinsic_matrix_, extrinsic, intrinsic.width_,
                    intrinsic.height_);
    }

    void SetupCamera(const Eigen::Matrix3d &intrinsic,
                     const Eigen::Matrix4d &extrinsic,
                     int intrinsic_width_px,
                     int intrinsic_height_px) {
        Camera::SetupCameraAsPinholeCamera(
                *scene_->GetCamera(), intrinsic, extrinsic, intrinsic_width_px,
                intrinsic_height_px, scene_->GetBoundingBox());
    }

    void SetupCamera(float verticalFoV,
                     const Eigen::Vector3f &center,
                     const Eigen::Vector3f &eye,
                     const Eigen::Vector3f &up) {
        float aspect = 1.0f;
        if (height_ > 0) {
            aspect = float(width_) / float(height_);
        }
        auto *camera = scene_->GetCamera();
        auto far_plane =
                Camera::CalcFarPlane(*camera, scene_->GetBoundingBox());
        camera->SetProjection(verticalFoV, aspect, Camera::CalcNearPlane(),
                              far_plane, rendering::Camera::FovType::Vertical);
        camera->LookAt(center, eye, up);
    }

private:
    int width_;
    int height_;
    FilamentRenderer *renderer_;
    // The offscreen renderer owns the scene so that it can clean it up
    // in the right order (otherwise we will crash).
    Open3DScene *scene_;
};

void pybind_rendering_classes(py::module &m) {
    py::class_<Renderer> renderer(
            m, "Renderer",
            "Renderer class that manages 3D resources. Get from gui.Window.");
    renderer.def("set_clear_color", &Renderer::SetClearColor,
                 "Sets the background color for the renderer, [r, g, b, a]. "
                 "Applies to everything being rendered, so it essentially acts "
                 "as the background color of the window")
            .def("add_texture",
                 (TextureHandle(Renderer::*)(
                         const std::shared_ptr<geometry::Image>, bool)) &
                         Renderer::AddTexture,
                 "image"_a, "is_sRGB"_a = false,
                 "Adds a texture: add_texture(geometry.Image, bool). The first "
                 "parameter is the image, the second parameter is optional and "
                 "is True if the image is in the sRGB colorspace and False "
                 "otherwise")
            .def("update_texture",
                 (bool (Renderer::*)(TextureHandle,
                                     const std::shared_ptr<geometry::Image>,
                                     bool)) &
                         Renderer::UpdateTexture,
                 "texture"_a, "image"_a, "is_sRGB"_a = false,
                 "Updates the contents of the texture to be the new image, or "
                 "returns False and does nothing if the image is a different "
                 "size. It is more efficient to call update_texture() rather "
                 "than removing and adding a new texture, especially when "
                 "changes happen frequently, such as when implmenting video. "
                 "add_texture(geometry.Image, bool). The first parameter is "
                 "the image, the second parameter is optional and is True "
                 "if the image is in the sRGB colorspace and False otherwise")
            .def("remove_texture", &Renderer::RemoveTexture,
                 "Deletes the texture. This does not remove the texture from "
                 "any existing materials or GUI widgets, and must be done "
                 "prior to this call.");

    // It would be nice to have this inherit from Renderer, but the problem is
    // that Python needs to own this class and Python needs to not own Renderer,
    // and pybind does not let us mix the two styles of ownership.
    py::class_<PyOffscreenRenderer, std::shared_ptr<PyOffscreenRenderer>>
            offscreen(m, "OffscreenRenderer",
                      "Renderer instance that can be used for rendering to an "
                      "image");
    offscreen
            .def(py::init([](int w, int h, const std::string &resource_path,
                             bool headless) {
                     return std::make_shared<PyOffscreenRenderer>(
                             w, h, resource_path, headless);
                 }),
                 "width"_a, "height"_a, "resource_path"_a = "",
                 "headless"_a = false,
                 "Takes width, height and optionally a resource_path and "
                 "headless flag. If "
                 "unspecified, resource_path will use the resource path from "
                 "the installed Open3D library. By default a running windowing "
                 "session is required. To enable headless rendering set "
                 "headless to True")
            .def_property_readonly(
                    "scene", &PyOffscreenRenderer::GetScene,
                    "Returns the Open3DScene for this renderer. This scene is "
                    "destroyed when the renderer is destroyed and should not "
                    "be accessed after that point.")
            .def("setup_camera",
                 py::overload_cast<float, const Eigen::Vector3f &,
                                   const Eigen::Vector3f &,
                                   const Eigen::Vector3f &>(
                         &PyOffscreenRenderer::SetupCamera),
                 "setup_camera(vertical_field_of_view, center, eye, up): "
                 "sets camera view using bounding box of current geometry")
            .def("setup_camera",
                 py::overload_cast<const camera::PinholeCameraIntrinsic &,
                                   const Eigen::Matrix4d &>(
                         &PyOffscreenRenderer::SetupCamera),
                 "setup_camera(intrinsics, extrinsic_matrix): "
                 "sets the camera view using bounding box of current geometry")
            .def("setup_camera",
                 py::overload_cast<const Eigen::Matrix3d &,
                                   const Eigen::Matrix4d &, int, int>(
                         &PyOffscreenRenderer::SetupCamera),
                 "setup_camera(intrinsic_matrix, extrinsic_matrix, "
                 "intrinsic_width_px, intrinsic_height_px): "
                 "sets the camera view using bounding box of current geometry")
            .def("render_to_image", &PyOffscreenRenderer::RenderToImage,
                 "Renders scene to an image, blocking until the image is "
                 "returned")
            .def("render_to_depth_image",
                 &PyOffscreenRenderer::RenderToDepthImage,
                 "Renders scene depth buffer to a float image, blocking until "
                 "the image is returned. Pixels range from 0 (near plane) to "
                 "1 (far plane)");

    // ---- Camera ----
    py::class_<Camera, std::shared_ptr<Camera>> cam(m, "Camera",
                                                    "Camera object");
    py::enum_<Camera::FovType> fov_type(cam, "FovType", py::arithmetic(),
                                        "Enum class for Camera field of view "
                                        "types.");
    fov_type.value("Vertical", Camera::FovType::Vertical)
            .value("Horizontal", Camera::FovType::Horizontal)
            .export_values();

    py::enum_<Camera::Projection> proj_type(cam, "Projection", py::arithmetic(),
                                            "Enum class for Camera projection "
                                            "types.");
    proj_type.value("Perspective", Camera::Projection::Perspective)
            .value("Ortho", Camera::Projection::Ortho)
            .export_values();

    cam.def("set_projection",
            (void (Camera::*)(double, double, double, double,
                              Camera::FovType)) &
                    Camera::SetProjection,
            "Sets a perspective projection. set_projection(field_of_view, "
            "aspect_ratio, far_plane, field_of_view_type)")
            .def("set_projection",
                 (void (Camera::*)(Camera::Projection, double, double, double,
                                   double, double, double)) &
                         Camera::SetProjection,
                 "Sets the camera projection via a viewing frustum. "
                 "set_projection(projection_type, left, right, bottom, top, "
                 "near, far)")
            .def("set_projection",
                 (void (Camera::*)(const Eigen::Matrix3d &, double, double,
                                   double, double)) &
                         Camera::SetProjection,
                 "Sets the camera projection via intrinsics matrix. "
                 "set_projection(intrinsics, near_place, far_plane, "
                 "image_width, image_height)")
            .def("look_at", &Camera::LookAt,
                 "Sets the position and orientation of the camera: "
                 "look_at(center, eye, up)")
            .def("unproject", &Camera::Unproject,
                 "unproject(x, y, z, view_width, view_height): takes the "
                 "(x, y, z) location in the view, where x, y are the number of "
                 "pixels from the upper left of the view, and z is the depth "
                 "value. Returns the world coordinate (x', y', z').")
            .def("copy_from", &Camera::CopyFrom,
                 "Copies the settings from the camera passed as the argument "
                 "into this camera")
            .def("get_near", &Camera::GetNear,
                 "Returns the distance from the camera to the near plane")
            .def("get_far", &Camera::GetFar,
                 "Returns the distance from the camera to the far plane")
            .def("get_field_of_view", &Camera::GetFieldOfView,
                 "Returns the field of view of camera, in degrees. Only valid "
                 "if it was passed to set_projection().")
            .def("get_field_of_view_type", &Camera::GetFieldOfViewType,
                 "Returns the field of view type. Only valid if it was passed "
                 "to set_projection().")
            .def(
                    "get_projection_matrix",
                    [](const Camera &cam) -> Eigen::Matrix4f {
                        // GetProjectionMatrix() returns Eigen::Transform which
                        // doesn't have a conversion to a Python object
                        return cam.GetProjectionMatrix().matrix();
                    },
                    "Returns the projection matrix of the camera")
            .def(
                    "get_view_matrix",
                    [](const Camera &cam) -> Eigen::Matrix4f {
                        return cam.GetViewMatrix().matrix();
                    },
                    "Returns the view matrix of the camera")
            .def(
                    "get_model_matrix",
                    [](const Camera &cam) -> Eigen::Matrix4f {
                        return cam.GetModelMatrix().matrix();
                    },
                    "Returns the model matrix of the camera");

    // ---- Gradient ----
    py::class_<Gradient, std::shared_ptr<Gradient>> gradient(
            m, "Gradient",
            "Manages a gradient for the unlitGradient shader."
            "In gradient mode, the array of points specifies points along "
            "the gradient, from 0 to 1 (inclusive). These do need to be "
            "evenly spaced."
            "Simple greyscale:"
            "    [ ( 0.0, black ),"
            "      ( 1.0, white ) ]"
            "Rainbow (note the gaps around green):"
            "    [ ( 0.000, blue ),"
            "      ( 0.125, cornflower blue ),"
            "      ( 0.250, cyan ),"
            "      ( 0.500, green ),"
            "      ( 0.750, yellow ),"
            "      ( 0.875, orange ),"
            "      ( 1.000, red ) ]"
            "The gradient will generate a largish texture, so it should "
            "be fairly smooth, but the boundaries may not be exactly as "
            "specified due to quantization imposed by the fixed size of "
            "the texture."
            "  The points *must* be sorted from the smallest value to the "
            "largest. The values must be in the range [0, 1].");
    py::enum_<Gradient::Mode> gradient_mode(gradient, "Mode", py::arithmetic());
    gradient_mode.value("GRADIENT", Gradient::Mode::kGradient)
            .value("LUT", Gradient::Mode::kLUT)
            .export_values();
    py::class_<Gradient::Point> gpt(gradient, "Point");
    gpt.def(py::init<float, const Eigen::Vector4f>())
            .def("__repr__",
                 [](const Gradient::Point &p) {
                     std::stringstream s;
                     s << "Gradient.Point[" << p.value << ", (" << p.color[0]
                       << ", " << p.color[1] << ", " << p.color[2] << ", "
                       << p.color[3] << ")]";
                     return s.str();
                 })
            .def_readwrite("value", &Gradient::Point::value,
                           "Must be within 0.0 and 1.0")
            .def_readwrite("color", &Gradient::Point::color,
                           "[R, G, B, A]. Color values must be in [0.0, 1.0]");
    gradient.def(py::init<>())
            .def(py::init<std::vector<Gradient::Point>>())
            .def_property("points", &Gradient::GetPoints, &Gradient::SetPoints)
            .def_property("mode", &Gradient::GetMode, &Gradient::SetMode);

    // ---- Material ----
    py::class_<Material> mat(m, "Material",
                             "Describes the real-world, physically based (PBR) "
                             "material used to render a geometry");
    mat.def(py::init<>())
            .def_readwrite("has_alpha", &Material::has_alpha)
            .def_readwrite("base_color", &Material::base_color)
            .def_readwrite("base_metallic", &Material::base_metallic)
            .def_readwrite("base_roughness", &Material::base_roughness)
            .def_readwrite("base_reflectance", &Material::base_reflectance)
            .def_readwrite("base_clearcoat", &Material::base_clearcoat)
            .def_readwrite("base_clearcoat_roughness",
                           &Material::base_clearcoat_roughness)
            .def_readwrite("base_anisotropy", &Material::base_anisotropy)
            .def_readwrite("thickness", &Material::thickness)
            .def_readwrite("transmission", &Material::transmission)
            .def_readwrite("absorption_color", &Material::absorption_color)
            .def_readwrite("absorption_distance",
                           &Material::absorption_distance)
            .def_readwrite("point_size", &Material::point_size)
            .def_readwrite("line_width", &Material::line_width,
                           "Requires 'shader' to be 'unlitLine'")
            .def_readwrite("albedo_img", &Material::albedo_img)
            .def_readwrite("normal_img", &Material::normal_img)
            .def_readwrite("ao_img", &Material::ao_img)
            .def_readwrite("metallic_img", &Material::metallic_img)
            .def_readwrite("roughness_img", &Material::roughness_img)
            .def_readwrite("reflectance_img", &Material::reflectance_img)
            .def_readwrite("clearcoat_img", &Material::clearcoat_img)
            .def_readwrite("clearcoat_roughness_img",
                           &Material::clearcoat_roughness_img)
            .def_readwrite("anisotropy_img", &Material::anisotropy_img)
            .def_readwrite("generic_params", &Material::generic_params)
            .def_readwrite("generic_imgs", &Material::generic_imgs)
            .def_readwrite("gradient", &Material::gradient)
            .def_readwrite("scalar_min", &Material::scalar_min)
            .def_readwrite("scalar_max", &Material::scalar_max)
            .def_readwrite("sRGB_color", &Material::sRGB_color)
            .def_readwrite("aspect_ratio", &Material::aspect_ratio)
            .def_readwrite("ground_plane_axis", &Material::ground_plane_axis)
            .def_readwrite("shader", &Material::shader);

    // ---- TriangleMeshModel ----
    py::class_<TriangleMeshModel, std::shared_ptr<TriangleMeshModel>> tri_model(
            m, "TriangleMeshModel",
            "A list of geometry.TriangleMesh and Material that can describe a "
            "complex model with multiple meshes, such as might be stored in an "
            "FBX, OBJ, or GLTF file");
    py::class_<TriangleMeshModel::MeshInfo> tri_model_info(tri_model,
                                                           "MeshInfo", "");
    tri_model_info
            .def(py::init([](std::shared_ptr<geometry::TriangleMesh> mesh,
                             const std::string &name,
                             unsigned int material_idx) {
                return TriangleMeshModel::MeshInfo{mesh, name, material_idx};
            }))
            .def_readwrite("mesh", &TriangleMeshModel::MeshInfo::mesh)
            .def_readwrite("mesh_name", &TriangleMeshModel::MeshInfo::mesh_name)
            .def_readwrite("material_idx",
                           &TriangleMeshModel::MeshInfo::material_idx);
    tri_model.def(py::init<>())
            .def_readwrite("meshes", &TriangleMeshModel::meshes_)
            .def_readwrite("materials", &TriangleMeshModel::materials_);

    // ---- ColorGradingParams ---
    py::class_<ColorGradingParams> color_grading(
            m, "ColorGrading", "Parameters to control color grading options");

    py::enum_<ColorGradingParams::Quality> cgp_quality(
            color_grading, "Quality",
            "Quality level of color grading operations");
    cgp_quality.value("LOW", ColorGradingParams::Quality::kLow)
            .value("MEDIUM", ColorGradingParams::Quality::kMedium)
            .value("HIGH", ColorGradingParams::Quality::kHigh)
            .value("ULTRA", ColorGradingParams::Quality::kUltra);

    py::enum_<ColorGradingParams::ToneMapping> cgp_tone(
            color_grading, "ToneMapping",
            "Specifies the tone-mapping algorithm");
    cgp_tone.value("LINEAR", ColorGradingParams::ToneMapping::kLinear)
            .value("ACES_LEGACY", ColorGradingParams::ToneMapping::kAcesLegacy)
            .value("ACES", ColorGradingParams::ToneMapping::kAces)
            .value("FILMIC", ColorGradingParams::ToneMapping::kFilmic)
            .value("UCHIMURA", ColorGradingParams::ToneMapping::kUchimura)
            .value("REINHARD", ColorGradingParams::ToneMapping::kReinhard)
            .value("DISPLAY_RANGE",
                   ColorGradingParams::ToneMapping::kDisplayRange);

    color_grading
            .def(py::init([](ColorGradingParams::Quality q,
                             ColorGradingParams::ToneMapping algorithm) {
                return ColorGradingParams(q, algorithm);
            }))
            .def_property("quality", &ColorGradingParams::GetQuality,
                          &ColorGradingParams::SetQuality,
                          "Quality of color grading operations. High quality "
                          "is more accurate but slower")
            .def_property("tone_mapping", &ColorGradingParams::GetToneMapping,
                          &ColorGradingParams::SetToneMapping,
                          "The tone mapping algorithm to apply. Must be one of "
                          "Linear, AcesLegacy, Aces, Filmic, Uchimura, "
                          "Rienhard, Display Range(for debug)")
            .def_property("temperature", &ColorGradingParams::GetTemperature,
                          &ColorGradingParams::SetTemperature,
                          "White balance color temperature")
            .def_property(
                    "tint", &ColorGradingParams::GetTint,
                    &ColorGradingParams::SetTint,
                    "Tint on the green/magenta axis. Ranges from -1.0 to 1.0.");

    // ---- View ----
    py::class_<View, UnownedPointer<View>> view(m, "View",
                                                "Low-level view class");
    view.def("set_color_grading", &View::SetColorGrading,
             "Sets the parameters to be used for the color grading algorithms");

    // ---- Scene ----
    py::class_<Scene, UnownedPointer<Scene>> scene(m, "Scene",
                                                   "Low-level rendering scene");
    py::enum_<Scene::GroundPlane> ground_plane(
            scene, "GroundPlane", py::arithmetic(),
            "Plane on which to show ground plane: XZ, XY, or YZ");
    ground_plane.value("XZ", Scene::GroundPlane::XZ)
            .value("XY", Scene::GroundPlane::XY)
            .value("YZ", Scene::GroundPlane::YZ)
            .export_values();
    scene.def("add_camera", &Scene::AddCamera, "Adds a camera to the scene")
            .def("remove_camera", &Scene::RemoveCamera,
                 "Removes the camera with the given name")
            .def("set_active_camera", &Scene::SetActiveCamera,
                 "Sets the camera with the given name as the active camera for "
                 "the scene")
            .def("add_geometry",
                 (bool (Scene::*)(
                         const std::string &, const geometry::Geometry3D &,
                         const Material &, const std::string &, size_t)) &
                         Scene::AddGeometry,
                 "name"_a, "geometry"_a, "material"_a,
                 "downsampled_name"_a = "", "downsample_threshold"_a = SIZE_MAX,
                 "Adds a Geometry with a material to the scene")
            .def("add_geometry",
                 (bool (Scene::*)(
                         const std::string &, const t::geometry::PointCloud &,
                         const Material &, const std::string &, size_t)) &
                         Scene::AddGeometry,
                 "name"_a, "geometry"_a, "material"_a,
                 "downsampled_name"_a = "", "downsample_threshold"_a = SIZE_MAX,
                 "Adds a Geometry with a material to the scene")
            .def("has_geometry", &Scene::HasGeometry,
                 "Returns True if a geometry with the provided name exists in "
                 "the scene.")
            .def("update_geometry", &Scene::UpdateGeometry,
                 "Updates the flagged arrays from the tgeometry.PointCloud. "
                 "The flags should be ORed from Scene.UPDATE_POINTS_FLAG, "
                 "Scene.UPDATE_NORMALS_FLAG, Scene.UPDATE_COLORS_FLAG, and "
                 "Scene.UPDATE_UV0_FLAG")
            .def("enable_indirect_light", &Scene::EnableIndirectLight,
                 "Enables or disables indirect lighting")
            .def("set_indirect_light", &Scene::SetIndirectLight,
                 "Loads the indirect light. The name parameter is the name of "
                 "the file to load")
            .def("set_indirect_light_intensity",
                 &Scene::SetIndirectLightIntensity,
                 "Sets the brightness of the indirect light")
            .def("enable_sun_light", &Scene::EnableSunLight)
            .def("set_sun_light", &Scene::SetSunLight,
                 "Sets the parameters of the sun light: direction, "
                 "color, intensity")
            .def("set_sun_light_direction", &Scene::SetSunLightDirection,
                 "Sets the sunlight direction.")
            .def("add_point_light", &Scene::AddPointLight,
                 "Adds a point light to the scene: add_point_light(name, "
                 "color, position, intensity, falloff, cast_shadows)")
            .def("add_spot_light", &Scene::AddSpotLight,
                 "Adds a spot light to the scene: add_point_light(name, "
                 "color, position, direction, intensity, falloff, "
                 "inner_cone_angle, outer_cone_angle, cast_shadows)")
            .def("add_directional_light", &Scene::AddDirectionalLight,
                 "Adds a directional light to the scene: add_point_light(name, "
                 "color, intensity, cast_shadows)")
            .def("remove_light", &Scene::RemoveLight,
                 "Removes the named light from the scene: remove_light(name)")
            .def("update_light_color", &Scene::UpdateLightColor,
                 "Changes a point, spot, or directional light's color")
            .def("update_light_position", &Scene::UpdateLightPosition,
                 "Changes a point or spot light's position")
            .def("update_light_direction", &Scene::UpdateLightDirection,
                 "Changes a spot or directional light's direction")
            .def("update_light_intensity", &Scene::UpdateLightIntensity,
                 "Changes a point, spot or directional light's intensity")
            .def("update_light_falloff", &Scene::UpdateLightFalloff,
                 "Changes a point or spot light's falloff")
            .def("update_light_cone_angles", &Scene::UpdateLightConeAngles,
                 "Changes a spot light's inner and outer cone angles")
            .def("enable_light_shadow", &Scene::EnableLightShadow,
                 "Changes whether a point, spot, or directional light can "
                 "cast shadows:  enable_light_shadow(name, can_cast_shadows)")
            .def("render_to_image", &Scene::RenderToImage,
                 "Renders the scene to an image. This can only be used in a "
                 "GUI app. To render without a window, use "
                 "Application.render_to_image")
            .def("render_to_depth_image", &Scene::RenderToDepthImage,
                 "Renders the scene to a depth image. This can only be used in "
                 "GUI app. To render without a window, use "
                 "Application.render_to_depth_image. Pixels range from "
                 "0.0 (near plane) to 1.0 (far plane)");

    scene.attr("UPDATE_POINTS_FLAG") = py::int_(Scene::kUpdatePointsFlag);
    scene.attr("UPDATE_NORMALS_FLAG") = py::int_(Scene::kUpdateNormalsFlag);
    scene.attr("UPDATE_COLORS_FLAG") = py::int_(Scene::kUpdateColorsFlag);
    scene.attr("UPDATE_UV0_FLAG") = py::int_(Scene::kUpdateUv0Flag);

    // ---- Open3DScene ----
    py::class_<Open3DScene, UnownedPointer<Open3DScene>> o3dscene(
            m, "Open3DScene", "High-level scene for rending");
    py::enum_<Open3DScene::LightingProfile> lighting(
            o3dscene, "LightingProfile", py::arithmetic(),
            "Enum for conveniently setting lighting");
    py::enum_<Open3DScene::UpDir> updir(
            o3dscene, "UpDir", py::arithmetic(),
            "Enum for specifying the up-direction of the model");
    lighting.value("HARD_SHADOWS", Open3DScene::LightingProfile::HARD_SHADOWS)
            .value("DARK_SHADOWS", Open3DScene::LightingProfile::DARK_SHADOWS)
            .value("MED_SHADOWS", Open3DScene::LightingProfile::MED_SHADOWS)
            .value("SOFT_SHADOWS", Open3DScene::LightingProfile::SOFT_SHADOWS)
            .value("NO_SHADOWS", Open3DScene::LightingProfile::NO_SHADOWS)
            .export_values();
    updir.value("PLUS_Y", Open3DScene::UpDir::PLUS_Y)
            .value("MINUS_Y", Open3DScene::UpDir::MINUS_Y)
            .value("PLUS_Z", Open3DScene::UpDir::PLUS_Z)
            .value("MINUS_Z", Open3DScene::UpDir::MINUS_Z);

    o3dscene.def(py::init<Renderer &>())
            .def("show_skybox", &Open3DScene::ShowSkybox,
                 "Toggles display of the skybox")
            .def("show_axes", &Open3DScene::ShowAxes,
                 "Toggles display of xyz axes")
            .def("show_ground_plane", &Open3DScene::ShowGroundPlane,
                 "Toggles display of ground plane")
            .def("set_lighting", &Open3DScene::SetLighting,
                 "Sets a simple lighting model. set_lighting(profile). "
                 "The default value is "
                 "Open3DScene.LightingProfile.MED_SHADOWS.")
            .def(
                    "set_background_color",
                    [](Open3DScene &scene, const Eigen::Vector4f &color) {
                        utility::LogWarning(
                                "visualization.rendering.Open3DScene.set_"
                                "background_color()\nhas been deprecated. "
                                "Please use set_background() instead.");
                        scene.SetBackground(color, nullptr);
                    },
                    "This function has been deprecated. Please use "
                    "set_background() instead.")
            .def("set_background", &Open3DScene::SetBackground, "color"_a,
                 "image"_a = nullptr,
                 "set_background([r, g, b, a], image=None). Sets the "
                 "background color and (optionally) image of the scene. ")
            .def("clear_geometry", &Open3DScene::ClearGeometry)
            .def("add_geometry",
                 py::overload_cast<const std::string &,
                                   const geometry::Geometry3D *,
                                   const Material &, bool>(
                         &Open3DScene::AddGeometry),
                 "name"_a, "geometry"_a, "material"_a,
                 "add_downsampled_copy_for_fast_rendering"_a = true)
            .def("add_geometry",
                 py::overload_cast<const std::string &,
                                   const t::geometry::PointCloud *,
                                   const Material &, bool>(
                         &Open3DScene::AddGeometry),
                 "name"_a, "geometry"_a, "material"_a,
                 "add_downsampled_copy_for_fast_rendering"_a = true)
            .def("add_model", &Open3DScene::AddModel,
                 "Adds TriangleMeshModel to the scene.")
            .def("has_geometry", &Open3DScene::HasGeometry,
                 "has_geometry(name): returns True if the geometry has been "
                 "added to the scene, False otherwise")
            .def("remove_geometry", &Open3DScene::RemoveGeometry,
                 "Removes the geometry with the given name")
            .def("modify_geometry_material",
                 &Open3DScene::ModifyGeometryMaterial,
                 "modify_geometry_material(name, material). Modifies the "
                 "material of the specified geometry")
            .def("show_geometry", &Open3DScene::ShowGeometry,
                 "Shows or hides the geometry with the given name")
            .def("update_material", &Open3DScene::UpdateMaterial,
                 "Applies the passed material to all the geometries")
            .def(
                    "set_view_size",
                    [](Open3DScene *scene, int width, int height) {
                        scene->GetView()->SetViewport(0, 0, width, height);
                    },
                    "Sets the view size. This should not be used except for "
                    "rendering to an image")
            .def_property_readonly("scene", &Open3DScene::GetScene,
                                   "The low-level rendering scene object "
                                   "(read-only)")
            .def_property_readonly("camera", &Open3DScene::GetCamera,
                                   "The camera object (read-only)")
            .def_property_readonly("bounding_box", &Open3DScene::GetBoundingBox,
                                   "The bounding box of all the items in the "
                                   "scene, visible and invisible")
            .def_property_readonly(
                    "view", &Open3DScene::GetView,
                    "The low level view associated with the scene")
            .def_property_readonly("background_color",
                                   &Open3DScene::GetBackgroundColor,
                                   "The background color (read-only)")
            .def_property("model_up", &Open3DScene::GetModelUp,
                          &Open3DScene::SetModelUp,
                          "Gets/sets the up-axis for the model. This affects "
                          "preset lighting directions and the ground plane")
            .def_property("downsample_threshold",
                          &Open3DScene::GetDownsampleThreshold,
                          &Open3DScene::SetDownsampleThreshold,
                          "Minimum number of points before downsampled point "
                          "clouds are created and used when rendering speed "
                          "is important");
}

void pybind_rendering(py::module &m) {
    py::module m_rendering = m.def_submodule("rendering");
    pybind_rendering_classes(m_rendering);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
