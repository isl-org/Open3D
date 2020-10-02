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
#include "open3d/visualization/rendering/Gradient.h"
#include "open3d/visualization/rendering/Material.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/Renderer.h"
#include "open3d/visualization/rendering/Scene.h"
#include "pybind/docstring.h"
#include "pybind/visualization/gui/gui.h"
#include "pybind11/functional.h"

namespace open3d {
namespace visualization {
namespace rendering {

void pybind_rendering_classes(py::module &m) {
    py::class_<Renderer> renderer(
            m, "Renderer",
            "Renderer class that manages 3D resources. Get from gui.Window.");
    renderer.def("set_clear_color", &Renderer::SetClearColor,
                 "Sets the background color for the renderer, [r, g, b, a]. "
                 "Applies to everything being rendered, so it essentially acts "
                 "as the background color of the window");

    // ---- Camera ----
    py::class_<Camera, std::shared_ptr<Camera>> cam(m, "Camera",
                                                    "Camera object");
    py::enum_<Camera::FovType> fov_type(cam, "FovType", py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    fov_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for Camera field of view types.";
            }),
            py::none(), py::none(), "");
    fov_type.value("Vertical", Camera::FovType::Vertical)
            .value("Horizontal", Camera::FovType::Horizontal)
            .export_values();

    py::enum_<Camera::Projection> proj_type(cam, "Projection",
                                            py::arithmetic());
    // Trick to write docs without listing the members in the enum class again.
    proj_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for Camera field of view types.";
            }),
            py::none(), py::none(), "");
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
            .def("look_at", &Camera::LookAt,
                 "Sets the position and orientation of the camera: "
                 "look_at(center, eye, up)");

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
            .def_readwrite("point_size", &Material::point_size)
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
            .def_readwrite("shader", &Material::shader);

    // ---- Scene ----
    py::class_<Scene, std::shared_ptr<Scene>> scene(
            m, "Scene", "Low-level rendering scene");
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
                 "The "
                 "flags should be ORed from Scene.UPDATE_POINTS_FLAG, "
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
            .def("enable_directional_light", &Scene::EnableDirectionalLight)
            .def("set_directional_light", &Scene::SetDirectionalLight,
                 "Sets the parameters of the directional light: direction, "
                 "color, intensity")
            .def("render_to_image", &Scene::RenderToImage,
                 "Renders the scene; image will be provided via a callback "
                 "function. The callback is necessary because rendering is "
                 "done on a different thread. The image remains valid "
                 "after the callback, assuming it was assigned somewhere.");
    scene.attr("UPDATE_POINTS_FLAG") = py::int_(Scene::kUpdatePointsFlag);
    scene.attr("UPDATE_NORMALS_FLAG") = py::int_(Scene::kUpdateNormalsFlag);
    scene.attr("UPDATE_COLORS_FLAG") = py::int_(Scene::kUpdateColorsFlag);
    scene.attr("UPDATE_UV0_FLAG") = py::int_(Scene::kUpdateUv0Flag);

    // ---- Open3DScene ----
    py::class_<Open3DScene, std::shared_ptr<Open3DScene>> o3dscene(
            m, "Open3DScene", "High-level scene for rending");
    o3dscene.def(py::init<Renderer &>())
            .def("show_skybox", &Open3DScene::ShowSkybox,
                 "Toggles display of the skybox")
            .def("show_axes", &Open3DScene::ShowAxes,
                 "Toggles display of xyz axes")
            .def("clear_geometry", &Open3DScene::ClearGeometry)
            .def("add_geometry",
                 py::overload_cast<const std::string &,
                                   std::shared_ptr<const geometry::Geometry3D>,
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
            .def("remove_geometry", &Open3DScene::RemoveGeometry,
                 "Removes the geometry with the given name")
            .def("show_geometry", &Open3DScene::ShowGeometry,
                 "Shows or hides the geometry with the given name")
            .def("update_material", &Open3DScene::UpdateMaterial,
                 "Applies the passed material to all the geometries")
            .def_property_readonly("scene", &Open3DScene::GetScene,
                                   "The low-level rendering scene object")
            .def_property_readonly("camera", &Open3DScene::GetCamera,
                                   "The camera object")
            .def_property_readonly("bounding_box", &Open3DScene::GetBoundingBox,
                                   "The bounding box of all the items in the "
                                   "scene, visible and invisible")
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
