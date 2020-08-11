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

#include "open3d/visualization/rendering/Material.h"
#include "open3d/visualization/rendering/Open3DScene.h"
#include "open3d/visualization/rendering/Renderer.h"
#include "open3d/visualization/rendering/Scene.h"
#include "pybind/docstring.h"
#include "pybind/visualization/gui/gui.h"
#include "pybind11/functional.h"

using namespace open3d::visualization;
using namespace open3d::visualization::rendering;

namespace open3d {

void pybind_rendering_classes(py::module &m) {
    py::class_<Renderer> renderer(
            m, "Renderer",
            "Renderer class that manages 3D resources. Get from gui.Window.");
    ;

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

    // ---- Material ----
    py::class_<Material> mat(m, "Material",
                             "Describes the real-world, physically based (PBR) "
                             "material used to render a geometry");
    mat.def(py::init<>())
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
                 (bool (Scene::*)(const std::string &,
                                  const geometry::Geometry3D &,
                                  const Material &)) &
                         Scene::AddGeometry,
                 "Adds a Geometry with a material to the scene")
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
                 "done "
                 "on a different thread. The image remains valid after the "
                 "callback, assuming it was assigned somewhere.");

    // ---- Open3DScene ----
    py::class_<Open3DScene, std::shared_ptr<Open3DScene>> o3dscene(
            m, "Open3DScene", "High-level scene for rending");
    o3dscene.def(py::init<Renderer &>())
            .def("show_skybox", &Open3DScene::ShowSkybox,
                 "Toggles display of the skybox")
            .def("show_axes", &Open3DScene::ShowAxes,
                 "Toggles display of xyz axes")
            .def("clear_geometry", &Open3DScene::ClearGeometry)
            .def("add_geometry", &Open3DScene::AddGeometry, "geometry"_a,
                 "material"_a,
                 "add_downsampled_copy_for_fast_rendering"_a = true)
            .def("update_material", &Open3DScene::UpdateMaterial,
                 "Applies the passed material to all the geometries")
            .def_property_readonly("scene", &Open3DScene::GetScene,
                                   "The low-level rendering scene object")
            .def_property_readonly("camera", &Open3DScene::GetCamera,
                                   "The camera object");
}

void pybind_rendering(py::module &m) {
    py::module m_rendering = m.def_submodule("rendering");
    pybind_rendering_classes(m_rendering);
}

}  // namespace open3d
