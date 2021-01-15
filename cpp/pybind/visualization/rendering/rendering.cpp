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

private:
    int width_;
    int height_;
    FilamentRenderer *renderer_;
    // The offscreen renderer owns the scene so that it can clean it up
    // in the right order (otherwise we will crash).
    Open3DScene *scene_;
};

void pybind_rendering_classes(py::module &m) {
    py::class_<Renderer> renderer(m, "Renderer",
                                  "The class provides methods to manage the "
                                  "global objects required to render an object."
                                  "You can use this class to manage material, "
                                  "texture, indirect light, and skybox.");
    renderer.def("set_clear_color", &Renderer::SetClearColor,
                 "Sets the background color for the renderer, "
                 "based on the [r, g, b, a] passed. "
                 "This applies BG color to everything being rendered, and acts"
                 "as the background color of the window.");

    // It would be nice to have this inherit from Renderer, but the problem is
    // that Python needs to own this class and Python needs to not own Renderer,
    // and pybind does not let us mix the two styls of ownership.
    py::class_<PyOffscreenRenderer, std::shared_ptr<PyOffscreenRenderer>>
            offscreen(m, "OffscreenRenderer",
                      "This class lets you render to an image. "
                      "This is useful if you want to save the "
                      "rendering to image.");
    offscreen
            .def(py::init([](int w, int h, const std::string &resource_path,
                             bool headless) {
                     return std::make_shared<PyOffscreenRenderer>(
                             w, h, resource_path, headless);
                 }),
                 "width"_a, "height"_a, "resource_path"_a = "",
                 "headless"_a = false,
                 "The method takes width, height and optionally a "
                 "resource_path and "
                 "headless flag. If resource_path is not specified, then"
                 " the installed Open3D library is used as resource_path. "
                 "By default a running windowing session is required. "
                 "To enable headless rendering, you must set "
                 "headless to True")
            .def_property_readonly(
                    "scene", &PyOffscreenRenderer::GetScene,
                    "This returns the Open3DScene for this renderer during "
                    "rendering."
                    " This is destructed with renderer, and andd should not be"
                    " accessed after that point.")
            .def("render_to_image", &PyOffscreenRenderer::RenderToImage,
                 "This renders scene to an image, and blocks until the image "
                 "is "
                 "returned");

    // ---- Camera ----
    py::class_<Camera, std::shared_ptr<Camera>> cam(
            m, "Camera",
            "This class provides methods"
            "that let you manage cameras in your rendered scene.");
    py::enum_<Camera::FovType> fov_type(
            cam, "FovType", py::arithmetic(),
            "The Enum class for camera's field of view types, "
            "and accepts these two values:");
    fov_type.value("Vertical", Camera::FovType::Vertical)
            .value("Horizontal", Camera::FovType::Horizontal)
            .export_values();

    py::enum_<Camera::Projection> proj_type(
            cam, "Projection", py::arithmetic(),
            "The enum class for camera's projection types, and accepts these "
            "two values:");
    proj_type.value("Perspective", Camera::Projection::Perspective)
            .value("Ortho", Camera::Projection::Ortho)
            .export_values();

    cam.def("set_projection",
            (void (Camera::*)(double, double, double, double,
                              Camera::FovType)) &
                    Camera::SetProjection,
            "The method sets the camera projection using a viewing frustum, "
            "and "
            "accepts projection_type, left, right, bottom, top, near, and far)")
            .def("set_projection",
                 (void (Camera::*)(Camera::Projection, double, double, double,
                                   double, double, double)) &
                         Camera::SetProjection,
                 "The method sets the camera projection through a viewing "
                 "frustum. "
                 "set_projection(projection_type, left, right, bottom, top, "
                 "near, far)")
            .def("set_projection",
                 (void (Camera::*)(const Eigen::Matrix3d &, double, double,
                                   double, double)) &
                         Camera::SetProjection,
                 "The method sets the camera projection via intrinsics matrix. "
                 "set_projection(intrinsics, near_place, far_plane, "
                 "image_width, image_height)")
            .def("look_at", &Camera::LookAt,
                 "The method sets the position and orientation of the camera: "
                 "look_at(center, eye, up)");
    // ---- Gradient ----
    py::class_<Gradient, std::shared_ptr<Gradient>> gradient(
            m, "Gradient",
            "The class provides methods to manage a gradient for the unlitGradient shader.
            In the Gradient mode,
            the array of points indicates the points along
            "
            "the gradient, from 0 to 1 (inclusive) and need to be evenly "
            "spaced."
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
            "The gradient generates a large texture, so it should "
            "be smooth. However, the boundaries may not be exactly as "
            "specified due to quantization imposed by the fixed size of "
            "the texture."
            "  The points *must* be sorted from the smallest value to the "
            "largest.");
    py::enum_<Gradient::Mode> gradient_mode(gradient, "Mode", py::arithmetic());
    gradient_mode.value("GRADIENT", Gradient::Mode::kGradient)
            .value("LUT", Gradient::Mode::kLUT)
            .export_values();
    py::class_<Gradient::Point> gpt(gradient, "Point",
                                    "Lets you get a point in the gradient.");
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
                           "The value must be within 0.0 and 1.0.")
            .def_readwrite("color", &Gradient::Point::color,
                           "The color in [R, G, B, A]. The values must be in "
                           "[0.0, 1.0].");
    gradient.def(py::init<>())
            .def(py::init<std::vector<Gradient::Point>>())
            .def_property("points", &Gradient::GetPoints, &Gradient::SetPoints)
            .def_property("mode", &Gradient::GetMode, &Gradient::SetMode);

    // ---- Material ----
    py::class_<Material> mat(m, "Material",
                             "The class provides methods to use real-world, "
                             "physically based (PBR) "
                             "material that can used to render a geometry."
                             "You can use one of the following materials:");
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
            .def_readwrite("line_width", &Material::line_width)
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
            .def_readwrite("shader", &Material::shader);

    // ---- TriangleMeshModel ----
    py::class_<TriangleMeshModel> tri_model(
            m, "TriangleMeshModel",
            "The class provides a list of geometry.TriangleMesh and Material "
            "that can"
            " describe a complex model with multiple meshes. These can be "
            "stored in an "
            "FBX, OBJ, or GLTF file.");
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
            m, "ColorGrading",
            "The class provides parameters to control color grading options.");
    color_grading
            .def(py::init([](ColorGradingParams::Quality q,
                             ColorGradingParams::ToneMapping algorithm) {
                return ColorGradingParams(q, algorithm);
            }))
            .def_property(
                    "quality", &ColorGradingParams::GetQuality,
                    &ColorGradingParams::SetQuality,
                    "The quality of color grading operations. High quality "
                    "is more accurate but slower")
            .def_property(
                    "tone_mapping", &ColorGradingParams::GetToneMapping,
                    &ColorGradingParams::SetToneMapping,
                    "The tone mapping algorithm to use, and must be one of "
                    "Linear, AcesLegacy, Aces, Filmic, Uchimura, "
                    "Rienhard, or Display Range(for debug).")
            .def_property("temperature", &ColorGradingParams::GetTemperature,
                          &ColorGradingParams::SetTemperature,
                          "The white balance color temperature to use.")
            .def_property("tint", &ColorGradingParams::GetTint,
                          &ColorGradingParams::SetTint,
                          "The tint on the green/magenta axis. Ranges from "
                          "-1.0 to 1.0.");

    // ---- View ----
    py::class_<View, UnownedPointer<View>> view(
            m, "View", "The class lets you render a low-level view.");
    view.def("set_color_grading", &View::SetColorGrading,
             "The method sets the parameters to be used for the color grading "
             "algorithms");

    // ---- Scene ----
    py::class_<Scene, UnownedPointer<Scene>> scene(m, "Scene",
                                                   "The low-level scene.");
    scene.def("add_camera", &Scene::AddCamera,
              "The method adds a camera to the scene.")
            .def("remove_camera", &Scene::RemoveCamera,
                 "The method removes the camera with the given name.")
            .def("set_active_camera", &Scene::SetActiveCamera,
                 "The method sets the camera with the given name as the active "
                 "camera for "
                 "the scene.")
            .def("add_geometry",
                 (bool (Scene::*)(
                         const std::string &, const geometry::Geometry3D &,
                         const Material &, const std::string &, size_t)) &
                         Scene::AddGeometry,
                 "name"_a, "geometry"_a, "material"_a,
                 "downsampled_name"_a = "", "downsample_threshold"_a = SIZE_MAX,
                 "The method adds a Geometry with a material to the scene.")
            .def("add_geometry",
                 (bool (Scene::*)(
                         const std::string &, const t::geometry::PointCloud &,
                         const Material &, const std::string &, size_t)) &
                         Scene::AddGeometry,
                 "name"_a, "geometry"_a, "material"_a,
                 "downsampled_name"_a = "", "downsample_threshold"_a = SIZE_MAX,
                 "The method adds a Geometry with a material to the scene.")
            .def("has_geometry", &Scene::HasGeometry,
                 "Returns True if a geometry with the provided name exists in "
                 "the scene.")
            .def("update_geometry", &Scene::UpdateGeometry,
                 "The method updates the flagged arrays from the "
                 "tgeometry.PointCloud. "
                 "The flags should be ORed from Scene.UPDATE_POINTS_FLAG, "
                 "Scene.UPDATE_NORMALS_FLAG, Scene.UPDATE_COLORS_FLAG, and "
                 "Scene.UPDATE_UV0_FLAG.")
            .def("enable_indirect_light", &Scene::EnableIndirectLight,
                 "The method enables or disables indirect lighting.")
            .def("set_indirect_light", &Scene::SetIndirectLight,
                 "The method loads the indirect light. The name parameter is "
                 "the name of "
                 "the file to load.")
            .def("set_indirect_light_intensity",
                 &Scene::SetIndirectLightIntensity,
                 "The method sets the brightness of the indirect light.")
            .def("enable_sun_light", &Scene::EnableSunLight,
                 "The method enables or "
                 "disables sun light.")
            .def("set_sun_light", &Scene::SetSunLight,
                 "The method sets the parameters of the sun light such as "
                 "direction, "
                 "color, or intensity.")
            .def("add_point_light", &Scene::AddPointLight,
                 "The method adds a point light to the scene: "
                 "add_point_light(name, "
                 "color, position, intensity, falloff, cast_shadows).")
            .def("add_spot_light", &Scene::AddSpotLight,
                 "The method adds a spot light to the scene: "
                 "add_point_light(name, "
                 "color, position, direction, intensity, falloff, "
                 "inner_cone_angle, outer_cone_angle, cast_shadows).")
            .def("add_directional_light", &Scene::AddDirectionalLight,
                 "The method adds a directional light to the scene: "
                 "add_point_light(name, "
                 "color, intensity, cast_shadows).")
            .def("remove_light", &Scene::RemoveLight,
                 "The method removes the named light from the scene: "
                 "remove_light(name).")
            .def("update_light_color", &Scene::UpdateLightColor,
                 "The method changes a point, spot, or directional light's "
                 "color.")
            .def("update_light_position", &Scene::UpdateLightPosition,
                 "The method changes a point or spot light's position.")
            .def("update_light_direction", &Scene::UpdateLightDirection,
                 "The method changes a spot or directional light's direction.")
            .def("update_light_intensity", &Scene::UpdateLightIntensity,
                 "The method changes a point, spot or directional light's "
                 "intensity.")
            .def("update_light_falloff", &Scene::UpdateLightFalloff,
                 "The method changes a point or spot light's falloff.")
            .def("update_light_cone_angles", &Scene::UpdateLightConeAngles,
                 "The method changes a spot light's inner and outer cone "
                 "angles.")
            .def("enable_light_shadow", &Scene::EnableLightShadow,
                 "The method changes whether a point, spot, or directional "
                 "light can "
                 "cast shadows:  enable_light_shadow(name, can_cast_shadows).")
            .def("render_to_image", &Scene::RenderToImage,
                 "The method renders the scene to an image. This can only be "
                 "used in a "
                 "GUI app. To render without a window, use "
                 "Application.render_to_image.");

    scene.attr("UPDATE_POINTS_FLAG") = py::int_(Scene::kUpdatePointsFlag);
    scene.attr("UPDATE_NORMALS_FLAG") = py::int_(Scene::kUpdateNormalsFlag);
    scene.attr("UPDATE_COLORS_FLAG") = py::int_(Scene::kUpdateColorsFlag);
    scene.attr("UPDATE_UV0_FLAG") = py::int_(Scene::kUpdateUv0Flag);

    // ---- Open3DScene ----
    py::class_<Open3DScene, UnownedPointer<Open3DScene>> o3dscene(
            m, "Open3DScene",
            "The class provides methods to manage a high-level rendering "
            "scene.");
    py::enum_<Open3DScene::LightingProfile> lighting(
            o3dscene, "LightingProfile", py::arithmetic(),
            "The enum for quickly setting the lighting.");
    lighting.value("HARD_SHADOWS", Open3DScene::LightingProfile::HARD_SHADOWS)
            .value("DARK_SHADOWS", Open3DScene::LightingProfile::DARK_SHADOWS)
            .value("MED_SHADOWS", Open3DScene::LightingProfile::MED_SHADOWS)
            .value("SOFT_SHADOWS", Open3DScene::LightingProfile::SOFT_SHADOWS)
            .value("NO_SHADOWS", Open3DScene::LightingProfile::NO_SHADOWS)
            .export_values();

    o3dscene.def(py::init<Renderer &>())
            .def("show_skybox", &Open3DScene::ShowSkybox,
                 "Lets you show or hide the skybox")
            .def("show_axes", &Open3DScene::ShowAxes,
                 "Lets you show or hide xyz axes.")
            .def("set_lighting", &Open3DScene::SetLighting,
                 "Lets you set a simple lighting model. set_lighting(profile, "
                 "sun_dir). The default value is "
                 "set_lighting(Open3DScene.LightingProfile.MED_SHADOWS, "
                 "(0.577, -0.577, -0.577))")
            .def(
                    "set_background_color",
                    [](Open3DScene &scene, const Eigen::Vector4f &color) {
                        utility::LogWarning(
                                "visualization.rendering.Open3DScene.set_"
                                "background_color() has been deprecated. "
                                "You must use set_background() instead.");
                        scene.SetBackground(color, nullptr);
                    },
                    "This function has been deprecated. You must use "
                    "set_background() instead.")
            .def("set_background", &Open3DScene::SetBackground, "color"_a,
                 "image"_a = nullptr,
                 "Lets you set the background color of the scene using [r, g, "
                 "b, a]."
                 " Sets the "
                 "background color and (optionally) image of the scene. ")
            .def("clear_geometry", &Open3DScene::ClearGeometry)
            .def("add_geometry",
                 py::overload_cast<const std::string &,
                                   const geometry::Geometry3D *,
                                   const Material &, bool>(
                         &Open3DScene::AddGeometry),
                 "name"_a, "geometry"_a, "material"_a,
                 "add_downsampled_copy_for_fast_rendering"_a = true,
                 "Lets you add a geometry using Geometry3D. You must pass the "
                 "scene name"
                 ", geometry, material, and specify if you want to add "
                 "downloaded sample for faster rendering.")
            .def("add_geometry",
                 py::overload_cast<const std::string &,
                                   const t::geometry::PointCloud *,
                                   const Material &, bool>(
                         &Open3DScene::AddGeometry),
                 "name"_a, "geometry"_a, "material"_a,
                 "add_downsampled_copy_for_fast_rendering"_a = true,
                 "Lets you add a geometry using PointCloud. You must pass the "
                 "scene name"
                 ", geometry, material, and specify if you want to add "
                 "downloaded sample for faster rendering.")
            .def("add_model", &Open3DScene::AddModel,
                 "Adds TriangleMeshModel to the scene.")
            .def("has_geometry", &Open3DScene::HasGeometry,
                 "has_geometry(name): returns True if the geometry has been "
                 "added to the scene, False otherwise")
            .def("remove_geometry", &Open3DScene::RemoveGeometry,
                 "Removes the geometry with the given name.")
            .def("modify_geometry_material",
                 &Open3DScene::ModifyGeometryMaterial,
                 "modify_geometry_material(name, material). Modifies the "
                 "material of the specified geometry.")
            .def("show_geometry", &Open3DScene::ShowGeometry,
                 "Shows or hides the geometry with the given name.")
            .def("update_material", &Open3DScene::UpdateMaterial,
                 "Applies the passed material to all the geometries.")
            .def(
                    "set_view_size",
                    [](Open3DScene *scene, int width, int height) {
                        scene->GetView()->SetViewport(0, 0, width, height);
                    },
                    "Sets the view size. This should not be used except for "
                    "rendering to an image")
            .def_property_readonly("scene", &Open3DScene::GetScene,
                                   "The low-level rendering scene object.")
            .def_property_readonly("camera", &Open3DScene::GetCamera,
                                   "The camera object used in the scene.")
            .def_property_readonly("bounding_box", &Open3DScene::GetBoundingBox,
                                   "The bounding box of all the items in the "
                                   "scene, visible and invisible.")
            .def_property_readonly(
                    "get_view", &Open3DScene::GetView,
                    "The low level view associated with the scene.")
            .def_property("downsample_threshold",
                          &Open3DScene::GetDownsampleThreshold,
                          &Open3DScene::SetDownsampleThreshold,
                          "Minimum number of points before downsampled point "
                          "clouds are created and used when rendering speed "
                          "is important.");
}

void pybind_rendering(py::module &m) {
    py::module m_rendering = m.def_submodule("rendering");
    pybind_rendering_classes(m_rendering);
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
