// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/camera/PinholeCameraIntrinsic.h"
#include "open3d/t/geometry/Geometry.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/visualization/rendering/ColorGrading.h"
#include "open3d/visualization/rendering/Gradient.h"
#include "open3d/visualization/rendering/MaterialRecord.h"
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
                        const std::string &resource_path) {
        gui::InitializeForPython(resource_path, true);
        width_ = width;
        height_ = height;
        renderer_ = new FilamentRenderer(EngineInstance::GetInstance(), width,
                                         height,
                                         EngineInstance::GetResourceManager());
        scene_ = new Open3DScene(*renderer_);
    }

    ~PyOffscreenRenderer() {
        delete scene_;
        delete renderer_;
        // Destroy Filament Engine here so OffscreenRenderer can be reused
        EngineInstance::DestroyInstance();
    }

    Open3DScene *GetScene() { return scene_; }

    std::shared_ptr<geometry::Image> RenderToImage() {
        return gui::RenderToImageWithoutWindow(scene_, width_, height_);
    }

    std::shared_ptr<geometry::Image> RenderToDepthImage(
            bool z_in_view_space = false) {
        return gui::RenderToDepthImageWithoutWindow(scene_, width_, height_,
                                                    z_in_view_space);
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
                     const Eigen::Vector3f &up,
                     float nearClip = -1.0f,
                     float farClip = -1.0f) {
        float aspect = 1.0f;
        if (height_ > 0) {
            aspect = float(width_) / float(height_);
        }
        auto *camera = scene_->GetCamera();
        auto far_plane = farClip > 0.0
                                 ? farClip
                                 : Camera::CalcFarPlane(
                                           *camera, scene_->GetBoundingBox());
        camera->SetProjection(
                verticalFoV, aspect,
                nearClip > 0.0 ? nearClip : Camera::CalcNearPlane(), far_plane,
                rendering::Camera::FovType::Vertical);
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

void pybind_rendering_declarations(py::module &m) {
    py::module m_rendering = m.def_submodule("rendering");
    py::class_<TextureHandle> texture_handle(m_rendering, "TextureHandle",
                                             "Handle to a texture");
    py::class_<Renderer> renderer(
            m_rendering, "Renderer",
            "Renderer class that manages 3D resources. Get from gui.Window.");
    // It would be nice to have this inherit from Renderer, but the problem is
    // that Python needs to own this class and Python needs to not own Renderer,
    // and pybind does not let us mix the two styles of ownership.
    py::class_<PyOffscreenRenderer, std::shared_ptr<PyOffscreenRenderer>>
            offscreen(m_rendering, "OffscreenRenderer",
                      "Renderer instance that can be used for rendering to an "
                      "image");
    py::class_<Camera, std::shared_ptr<Camera>> cam(m_rendering, "Camera",
                                                    "Camera object");
    py::native_enum<Camera::FovType>(
            cam, "FovType", "enum.Enum",
            "Enum class for Camera field of view types.")
            .value("Vertical", Camera::FovType::Vertical)
            .value("Horizontal", Camera::FovType::Horizontal)
            .export_values()
            .finalize();

    py::native_enum<Camera::Projection>(
            cam, "Projection", "enum.Enum",
            "Enum class for Camera projection types.")
            .value("Perspective", Camera::Projection::Perspective)
            .value("Ortho", Camera::Projection::Ortho)
            .export_values()
            .finalize();
    py::class_<Gradient, std::shared_ptr<Gradient>> gradient(
            m_rendering, "Gradient",
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
    py::native_enum<Gradient::Mode>(gradient, "Mode", "enum.Enum")
            .value("GRADIENT", Gradient::Mode::kGradient)
            .value("LUT", Gradient::Mode::kLUT)
            .export_values()
            .finalize();
    py::class_<Gradient::Point> gpt(gradient, "Point");
    py::class_<MaterialRecord> mat(
            m_rendering, "MaterialRecord",
            "Describes the real-world, physically based (PBR) "
            "material used to render a geometry");
    py::class_<TriangleMeshModel, std::shared_ptr<TriangleMeshModel>> tri_model(
            m_rendering, "TriangleMeshModel",
            "A list of geometry.TriangleMesh and Material that can describe a "
            "complex model with multiple meshes, such as might be stored in an "
            "FBX, OBJ, or GLTF file");
    py::class_<TriangleMeshModel::MeshInfo> tri_model_info(tri_model,
                                                           "MeshInfo", "");
    py::class_<ColorGradingParams> color_grading(
            m_rendering, "ColorGrading",
            "Parameters to control color grading options");
    py::native_enum<ColorGradingParams::Quality>(
            color_grading, "Quality", "enum.Enum",
            "Quality level of color grading operations")
            .value("LOW", ColorGradingParams::Quality::kLow)
            .value("MEDIUM", ColorGradingParams::Quality::kMedium)
            .value("HIGH", ColorGradingParams::Quality::kHigh)
            .value("ULTRA", ColorGradingParams::Quality::kUltra)
            .finalize();
    py::native_enum<ColorGradingParams::ToneMapping>(
            color_grading, "ToneMapping", "enum.Enum",
            "Specifies the tone-mapping algorithm")
            .value("LINEAR", ColorGradingParams::ToneMapping::kLinear)
            .value("ACES_LEGACY", ColorGradingParams::ToneMapping::kAcesLegacy)
            .value("ACES", ColorGradingParams::ToneMapping::kAces)
            .value("FILMIC", ColorGradingParams::ToneMapping::kFilmic)
            .value("UCHIMURA", ColorGradingParams::ToneMapping::kUchimura)
            .value("REINHARD", ColorGradingParams::ToneMapping::kReinhard)
            .value("DISPLAY_RANGE",
                   ColorGradingParams::ToneMapping::kDisplayRange)
            .finalize();
    py::class_<View, UnownedPointer<View>> view(m_rendering, "View",
                                                "Low-level view class");
    py::native_enum<View::ShadowType>(
            view, "ShadowType", "enum.Enum",
            "Available shadow mapping algorithm options")
            .value("PCF", View::ShadowType::kPCF)
            .value("VSM", View::ShadowType::kVSM)
            .finalize();
    py::class_<Scene, UnownedPointer<Scene>> scene(m_rendering, "Scene",
                                                   "Low-level rendering scene");
    py::native_enum<Scene::GroundPlane>(
            scene, "GroundPlane", "enum.Enum",
            "Plane on which to show ground plane: XZ, XY, or YZ")
            .value("XZ", Scene::GroundPlane::XZ)
            .value("XY", Scene::GroundPlane::XY)
            .value("YZ", Scene::GroundPlane::YZ)
            .export_values()
            .finalize();
    py::class_<Open3DScene, UnownedPointer<Open3DScene>> o3dscene(
            m_rendering, "Open3DScene", "High-level scene for rending");
    py::native_enum<Open3DScene::LightingProfile>(
            o3dscene, "LightingProfile", "enum.Enum",
            "Enum for conveniently setting lighting")
            .value("HARD_SHADOWS", Open3DScene::LightingProfile::HARD_SHADOWS)
            .value("DARK_SHADOWS", Open3DScene::LightingProfile::DARK_SHADOWS)
            .value("MED_SHADOWS", Open3DScene::LightingProfile::MED_SHADOWS)
            .value("SOFT_SHADOWS", Open3DScene::LightingProfile::SOFT_SHADOWS)
            .value("NO_SHADOWS", Open3DScene::LightingProfile::NO_SHADOWS)
            .export_values()
            .finalize();
}
void pybind_rendering_definitions(py::module &m) {
    auto m_rendering = static_cast<py::module>(m.attr("rendering"));
    auto renderer =
            static_cast<py::class_<Renderer>>(m_rendering.attr("Renderer"));
    renderer.def("set_clear_color", &Renderer::SetClearColor,
                 "Sets the background color for the renderer, [r, g, b, a]. "
                 "Applies to everything being rendered, so it essentially acts "
                 "as the background color of the window")
            .def("add_texture",
                 (TextureHandle(Renderer::*)(
                         const std::shared_ptr<geometry::Image>, bool)) &
                         Renderer::AddTexture,
                 "image"_a, "is_sRGB"_a = false,
                 "Adds a texture. The first parameter is the image, the second "
                 "parameter is optional and is True if the image is in the "
                 "sRGB colorspace and False otherwise")
            .def("update_texture",
                 (bool(Renderer::*)(TextureHandle,
                                    const std::shared_ptr<geometry::Image>,
                                    bool)) &
                         Renderer::UpdateTexture,
                 "texture"_a, "image"_a, "is_sRGB"_a = false,
                 "Updates the contents of the texture to be the new image, or "
                 "returns False and does nothing if the image is a different "
                 "size. It is more efficient to call update_texture() rather "
                 "than removing and adding a new texture, especially when "
                 "changes happen frequently, such as when implementing video. "
                 "add_texture(geometry.Image, bool). The first parameter is "
                 "the image, the second parameter is optional and is True "
                 "if the image is in the sRGB colorspace and False otherwise")
            .def("remove_texture", &Renderer::RemoveTexture, "texture"_a,
                 "Deletes the texture. This does not remove the texture from "
                 "any existing materials or GUI widgets, and must be done "
                 "prior to this call.");
    auto offscreen =
            static_cast<py::class_<PyOffscreenRenderer,
                                   std::shared_ptr<PyOffscreenRenderer>>>(
                    m_rendering.attr("OffscreenRenderer"));
    offscreen
            .def(py::init([](int w, int h, const std::string &resource_path) {
                     return std::make_shared<PyOffscreenRenderer>(
                             w, h, resource_path);
                 }),
                 "width"_a, "height"_a, "resource_path"_a = "",
                 "Takes width, height and optionally a resource_path. "
                 " If unspecified, resource_path will use the resource path "
                 "from "
                 "the installed Open3D library.")
            .def_property_readonly(
                    "scene", &PyOffscreenRenderer::GetScene,
                    "Returns the Open3DScene for this renderer. This scene is "
                    "destroyed when the renderer is destroyed and should not "
                    "be accessed after that point.")
            .def("setup_camera",
                 py::overload_cast<float, const Eigen::Vector3f &,
                                   const Eigen::Vector3f &,
                                   const Eigen::Vector3f &, float, float>(
                         &PyOffscreenRenderer::SetupCamera),
                 "vertical_field_of_view"_a, "center"_a, "eye"_a, "up"_a,
                 "near_clip"_a = -1.0f, "far_clip"_a = -1.0f,
                 "Sets camera view using bounding box of current geometry if "
                 "the near_clip and far_clip parameters are not set")
            .def("setup_camera",
                 py::overload_cast<const camera::PinholeCameraIntrinsic &,
                                   const Eigen::Matrix4d &>(
                         &PyOffscreenRenderer::SetupCamera),
                 "intrinsics"_a, "extrinsic_matrix"_a,
                 "Sets the camera view using bounding box of current geometry")
            .def("setup_camera",
                 py::overload_cast<const Eigen::Matrix3d &,
                                   const Eigen::Matrix4d &, int, int>(
                         &PyOffscreenRenderer::SetupCamera),
                 "intrinsic_matrix"_a, "extrinsic_matrix"_a,
                 "intrinsic_width_px"_a, "intrinsic_height_px"_a,
                 "Sets the camera view using bounding box of current geometry")
            .def("render_to_image", &PyOffscreenRenderer::RenderToImage,
                 "Renders scene to an image, blocking until the image is "
                 "returned")
            .def("render_to_depth_image",
                 &PyOffscreenRenderer::RenderToDepthImage,
                 "z_in_view_space"_a = false,
                 "Renders scene depth buffer to a float image, blocking until "
                 "the image is returned. Pixels range from 0 (near plane) to "
                 "1 (far plane). If z_in_view_space is set to True then pixels "
                 "are pre-transformed into view space (i.e., distance from "
                 "camera).");

    // ---- Camera ----
    auto cam = static_cast<py::class_<Camera, std::shared_ptr<Camera>>>(
            m_rendering.attr("Camera"));
    cam.def("set_projection",
            (void(Camera::*)(double, double, double, double, Camera::FovType)) &
                    Camera::SetProjection,
            "field_of_view"_a, "aspect_ratio"_a, "near_plane"_a, "far_plane"_a,
            "field_of_view_type"_a, "Sets a perspective projection.")
            .def("set_projection",
                 (void(Camera::*)(Camera::Projection, double, double, double,
                                  double, double, double)) &
                         Camera::SetProjection,
                 "projection_type"_a, "left"_a, "right"_a, "bottom"_a, "top"_a,
                 "near"_a, "far"_a,
                 "Sets the camera projection via a viewing frustum. ")
            .def("set_projection",
                 (void(Camera::*)(const Eigen::Matrix3d &, double, double,
                                  double, double)) &
                         Camera::SetProjection,
                 "intrinsics"_a, "near_plane"_a, "far_plane"_a, "image_width"_a,
                 "image_height"_a,
                 "Sets the camera projection via intrinsics matrix.")
            .def("look_at", &Camera::LookAt, "center"_a, "eye"_a, "up"_a,
                 "Sets the position and orientation of the camera: ")
            .def("unproject", &Camera::Unproject, "x"_a, "y"_a, "z"_a,
                 "view_width"_a, "view_height"_a,
                 "Takes the (x, y, z) location in the view, where x, y are the "
                 "number of pixels from the upper left of the view, and z is "
                 "the depth value. Returns the world coordinate (x', y', z').")
            .def("copy_from", &Camera::CopyFrom, "camera"_a,
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
    auto gradient =
            static_cast<py::class_<Gradient, std::shared_ptr<Gradient>>>(
                    m_rendering.attr("Gradient"));
    auto gpt = static_cast<py::class_<Gradient::Point>>(gradient.attr("Point"));
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
    auto mat = static_cast<py::class_<MaterialRecord>>(
            m_rendering.attr("MaterialRecord"));
    mat.def(py::init<>())
            .def_readwrite("has_alpha", &MaterialRecord::has_alpha)
            .def_readwrite("base_color", &MaterialRecord::base_color)
            .def_readwrite("base_metallic", &MaterialRecord::base_metallic)
            .def_readwrite("base_roughness", &MaterialRecord::base_roughness)
            .def_readwrite("base_reflectance",
                           &MaterialRecord::base_reflectance)
            .def_readwrite("base_clearcoat", &MaterialRecord::base_clearcoat)
            .def_readwrite("base_clearcoat_roughness",
                           &MaterialRecord::base_clearcoat_roughness)
            .def_readwrite("base_anisotropy", &MaterialRecord::base_anisotropy)
            .def_readwrite("emissive_color", &MaterialRecord::emissive_color)
            .def_readwrite("thickness", &MaterialRecord::thickness)
            .def_readwrite("transmission", &MaterialRecord::transmission)
            .def_readwrite("absorption_color",
                           &MaterialRecord::absorption_color)
            .def_readwrite("absorption_distance",
                           &MaterialRecord::absorption_distance)
            .def_readwrite("point_size", &MaterialRecord::point_size)
            .def_readwrite("line_width", &MaterialRecord::line_width,
                           "Requires 'shader' to be 'unlitLine'")
            .def_readwrite("albedo_img", &MaterialRecord::albedo_img)
            .def_readwrite("normal_img", &MaterialRecord::normal_img)
            .def_readwrite("ao_img", &MaterialRecord::ao_img)
            .def_readwrite("metallic_img", &MaterialRecord::metallic_img)
            .def_readwrite("roughness_img", &MaterialRecord::roughness_img)
            .def_readwrite("reflectance_img", &MaterialRecord::reflectance_img)
            .def_readwrite("clearcoat_img", &MaterialRecord::clearcoat_img)
            .def_readwrite("clearcoat_roughness_img",
                           &MaterialRecord::clearcoat_roughness_img)
            .def_readwrite("anisotropy_img", &MaterialRecord::anisotropy_img)
            .def_readwrite("ao_rough_metal_img",
                           &MaterialRecord::ao_rough_metal_img)
            .def_readwrite("generic_params", &MaterialRecord::generic_params)
            .def_readwrite("generic_imgs", &MaterialRecord::generic_imgs)
            .def_readwrite("gradient", &MaterialRecord::gradient)
            .def_readwrite("scalar_min", &MaterialRecord::scalar_min)
            .def_readwrite("scalar_max", &MaterialRecord::scalar_max)
            .def_readwrite("sRGB_color", &MaterialRecord::sRGB_color)
            .def_readwrite("aspect_ratio", &MaterialRecord::aspect_ratio)
            .def_readwrite("ground_plane_axis",
                           &MaterialRecord::ground_plane_axis)
            .def_readwrite("shader", &MaterialRecord::shader);

    // ---- TriangleMeshModel ----
    auto tri_model = static_cast<
            py::class_<TriangleMeshModel, std::shared_ptr<TriangleMeshModel>>>(
            m_rendering.attr("TriangleMeshModel"));
    auto tri_model_info = static_cast<py::class_<TriangleMeshModel::MeshInfo>>(
            tri_model.attr("MeshInfo"));
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
    auto color_grading = static_cast<py::class_<ColorGradingParams>>(
            m_rendering.attr("ColorGrading"));
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
    auto view = static_cast<py::class_<View, UnownedPointer<View>>>(
            m_rendering.attr("View"));
    view.def("set_color_grading", &View::SetColorGrading,
             "Sets the parameters to be used for the color grading algorithms")
            .def("set_post_processing", &View::SetPostProcessing,
                 "True to enable, False to disable post processing. Post "
                 "processing effects include: color grading, ambient occlusion "
                 "(and other screen space effects), and anti-aliasing.")
            .def("set_ambient_occlusion", &View::SetAmbientOcclusion,
                 "enabled"_a, "ssct_enabled"_a = false,
                 "True to enable, False to disable ambient occlusion. "
                 "Optionally, screen-space cone tracing may be enabled with "
                 "ssct_enabled=True.")
            .def("set_antialiasing", &View::SetAntiAliasing, "enabled"_a,
                 "temporal"_a = false,
                 "True to enable, False to disable anti-aliasing. Note that "
                 "this only impacts anti-aliasing post-processing. MSAA is "
                 "controlled separately by `set_sample_count`. Temporal "
                 "anti-aliasing may be optionally enabled with temporal=True.")
            .def("set_sample_count", &View::SetSampleCount,
                 "Sets the sample count for MSAA. Set to 1 to disable MSAA. "
                 "Typical values are 2, 4 or 8. The maximum possible value "
                 "depends on the underlying GPU and OpenGL driver.")
            .def("set_shadowing", &View::SetShadowing, "enabled"_a,
                 py::arg_v(
                         "type", View::ShadowType::kPCF,
                         "open3d.visualization.rendering.View.ShadowType.PCF"),
                 "True to enable, false to enable all shadow mapping when "
                 "rendering this View. When enabling shadow mapping you may "
                 "also specify one of two shadow mapping algorithms: PCF "
                 "(default) or VSM. Note: shadowing is enabled by default with "
                 "PCF shadow mapping.")
            .def("get_camera", &View::GetCamera,
                 "Returns the Camera associated with this View.");

    // ---- Scene ----
    auto scene = static_cast<py::class_<Scene, UnownedPointer<Scene>>>(
            m_rendering.attr("Scene"));
    scene.def("add_camera", &Scene::AddCamera, "name"_a, "camera"_a,
              "Adds a camera to the scene")
            .def("remove_camera", &Scene::RemoveCamera, "name"_a,
                 "Removes the camera with the given name")
            .def("set_active_camera", &Scene::SetActiveCamera, "name"_a,
                 "Sets the camera with the given name as the active camera for "
                 "the scene")
            .def("add_geometry",
                 (bool(Scene::*)(
                         const std::string &, const geometry::Geometry3D &,
                         const MaterialRecord &, const std::string &, size_t)) &
                         Scene::AddGeometry,
                 "name"_a, "geometry"_a, "material"_a,
                 "downsampled_name"_a = "", "downsample_threshold"_a = SIZE_MAX,
                 "Adds a Geometry with a material to the scene")
            .def("add_geometry",
                 (bool(Scene::*)(
                         const std::string &, const t::geometry::Geometry &,
                         const MaterialRecord &, const std::string &, size_t)) &
                         Scene::AddGeometry,
                 "name"_a, "geometry"_a, "material"_a,
                 "downsampled_name"_a = "", "downsample_threshold"_a = SIZE_MAX,
                 "Adds a Geometry with a material to the scene")
            .def("has_geometry", &Scene::HasGeometry, "name"_a,
                 "Returns True if a geometry with the provided name exists in "
                 "the scene.")
            .def("update_geometry", &Scene::UpdateGeometry, "name"_a,
                 "point_cloud"_a, "update_flag"_a,
                 "Updates the flagged arrays from the tgeometry.PointCloud. "
                 "The flags should be ORed from Scene.UPDATE_POINTS_FLAG, "
                 "Scene.UPDATE_NORMALS_FLAG, Scene.UPDATE_COLORS_FLAG, and "
                 "Scene.UPDATE_UV0_FLAG")
            .def("remove_geometry", &Scene::RemoveGeometry, "name"_a,
                 "Removes the named geometry from the scene.")
            .def("show_geometry", &Scene::ShowGeometry, "name"_a, "show"_a,
                 "Show or hide the named geometry.")
            .def("geometry_is_visible", &Scene::GeometryIsVisible, "name"_a,
                 "Returns false if the geometry is hidden, True otherwise. "
                 "Note: this is different from whether or not the geometry is "
                 "in view.")
            .def("geometry_shadows", &Scene::GeometryShadows, "name"_a,
                 "cast_shadows"_a, "receive_shadows"_a,
                 "Controls whether an object casts and/or receives shadows: "
                 "geometry_shadows(name, cast_shadows, receieve_shadows)")
            .def("set_geometry_culling", &Scene::SetGeometryCulling, "name"_a,
                 "enable"_a,
                 "Enable/disable view frustum culling on the named object. "
                 "Culling is enabled by default.")
            .def("set_geometry_priority", &Scene::SetGeometryPriority, "name"_a,
                 "priority"_a,
                 "Set sorting priority for named object. Objects with higher "
                 "priority will be rendering on top of overlapping geometry "
                 "with lower priority.")
            .def("enable_indirect_light", &Scene::EnableIndirectLight,
                 "enable"_a, "Enables or disables indirect lighting")
            .def("set_indirect_light", &Scene::SetIndirectLight, "name"_a,
                 "Loads the indirect light. The name parameter is the name of "
                 "the file to load")
            .def("set_indirect_light_intensity",
                 &Scene::SetIndirectLightIntensity, "intensity"_a,
                 "Sets the brightness of the indirect light")
            .def("enable_sun_light", &Scene::EnableSunLight, "enable"_a)
            .def("set_sun_light", &Scene::SetSunLight, "direction"_a, "color"_a,
                 "intensity"_a,
                 "Sets the parameters of the sun light direction, "
                 "color, intensity")
            .def("add_point_light", &Scene::AddPointLight, "name"_a, "color"_a,
                 "position"_a, "intensity"_a, "falloff"_a, "cast_shadows"_a,
                 "Adds a point light to the scene.")
            .def("add_spot_light", &Scene::AddSpotLight, "name"_a, "color"_a,
                 "position"_a, "direction"_a, "intensity"_a, "falloff"_a,
                 "inner_cone_angle"_a, "outer_cone_angle"_a, "cast_shadows"_a,
                 "Adds a spot light to the scene.")
            .def("add_directional_light", &Scene::AddDirectionalLight, "name"_a,
                 "color"_a, "direction"_a, "intensity"_a, "cast_shadows"_a,
                 "Adds a directional light to the scene")
            .def("remove_light", &Scene::RemoveLight, "name"_a,
                 "Removes the named light from the scene.")
            .def("update_light_color", &Scene::UpdateLightColor, "name"_a,
                 "color"_a,
                 "Changes a point, spot, or directional light's color")
            .def("update_light_position", &Scene::UpdateLightPosition, "name"_a,
                 "position"_a, "Changes a point or spot light's position.")
            .def("update_light_direction", &Scene::UpdateLightDirection,
                 "name"_a, "direction"_a,
                 "Changes a spot or directional light's direction.")
            .def("update_light_intensity", &Scene::UpdateLightIntensity,
                 "name"_a, "intensity"_a,
                 "Changes a point, spot or directional light's intensity.")
            .def("update_light_falloff", &Scene::UpdateLightFalloff, "name"_a,
                 "falloff"_a, "Changes a point or spot light's falloff.")
            .def("update_light_cone_angles", &Scene::UpdateLightConeAngles,
                 "name"_a, "inner_cone_angle"_a, "outer_cone_angle"_a,
                 "Changes a spot light's inner and outer cone angles.")
            .def("enable_light_shadow", &Scene::EnableLightShadow, "name"_a,
                 "can_cast_shadows"_a,
                 "Changes whether a point, spot, or directional light can "
                 "cast shadows.")
            .def("render_to_image", &Scene::RenderToImage,
                 "Renders the scene to an image. This can only be used in a "
                 "GUI app. To render without a window, use "
                 "``Application.render_to_image``.")
            .def("render_to_depth_image", &Scene::RenderToDepthImage,
                 "Renders the scene to a depth image. This can only be used in "
                 "GUI app. To render without a window, use "
                 "``Application.render_to_depth_image``. Pixels range from "
                 "0.0 (near plane) to 1.0 (far plane)");

    scene.attr("UPDATE_POINTS_FLAG") = py::int_(Scene::kUpdatePointsFlag);
    scene.attr("UPDATE_NORMALS_FLAG") = py::int_(Scene::kUpdateNormalsFlag);
    scene.attr("UPDATE_COLORS_FLAG") = py::int_(Scene::kUpdateColorsFlag);
    scene.attr("UPDATE_UV0_FLAG") = py::int_(Scene::kUpdateUv0Flag);

    // ---- Open3DScene ----
    auto o3dscene =
            static_cast<py::class_<Open3DScene, UnownedPointer<Open3DScene>>>(
                    m_rendering.attr("Open3DScene"));
    o3dscene.def(py::init<Renderer &>())
            .def("show_skybox", &Open3DScene::ShowSkybox, "enable"_a,
                 "Toggles display of the skybox")
            .def("show_axes", &Open3DScene::ShowAxes, "enable"_a,
                 "Toggles display of xyz axes")
            .def("show_ground_plane", &Open3DScene::ShowGroundPlane, "enable"_a,
                 "plane"_a, "Toggles display of ground plane")
            .def("set_lighting", &Open3DScene::SetLighting, "profile"_a,
                 "sun_dir"_a,
                 "Sets a simple lighting model. The default value is "
                 "set_lighting(Open3DScene.LightingProfile.MED_SHADOWS, "
                 "(0.577, -0.577, -0.577))")
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
                                   const MaterialRecord &, bool>(
                         &Open3DScene::AddGeometry),
                 "name"_a, "geometry"_a, "material"_a,
                 "add_downsampled_copy_for_fast_rendering"_a = true,
                 "Adds a geometry with the specified name. Default visible is "
                 "true.")
            .def("add_geometry",
                 py::overload_cast<const std::string &,
                                   const t::geometry::Geometry *,
                                   const MaterialRecord &, bool>(
                         &Open3DScene::AddGeometry),
                 "name"_a, "geometry"_a, "material"_a,
                 "add_downsampled_copy_for_fast_rendering"_a = true,
                 "Adds a geometry with the specified name. Default visible is "
                 "true.")
            .def("add_model", &Open3DScene::AddModel, "name"_a, "model"_a,
                 "Adds TriangleMeshModel to the scene.")
            .def("has_geometry", &Open3DScene::HasGeometry, "name"_a,
                 "Returns True if the geometry has been added to the scene, "
                 "False otherwise")
            .def("remove_geometry", &Open3DScene::RemoveGeometry, "name"_a,
                 "Removes the geometry with the given name")
            .def("geometry_is_visible", &Open3DScene::GeometryIsVisible,
                 "name"_a, "Returns True if the geometry name is visible")
            .def("set_geometry_transform", &Open3DScene::SetGeometryTransform,
                 "name"_a, "transform"_a,
                 "sets the pose of the geometry name to transform")
            .def("get_geometry_transform", &Open3DScene::GetGeometryTransform,
                 "name"_a, "Returns the pose of the geometry name in the scene")
            .def("modify_geometry_material",
                 &Open3DScene::ModifyGeometryMaterial, "name"_a, "material"_a,
                 "Modifies the material of the specified geometry")
            .def("show_geometry", &Open3DScene::ShowGeometry, "name"_a,
                 "show"_a, "Shows or hides the geometry with the given name")
            .def("update_material", &Open3DScene::UpdateMaterial, "material"_a,
                 "Applies the passed material to all the geometries")
            .def(
                    "set_view_size",
                    [](Open3DScene *scene, int width, int height) {
                        scene->GetView()->SetViewport(0, 0, width, height);
                    },
                    "width"_a, "height"_a,
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
            .def_property("downsample_threshold",
                          &Open3DScene::GetDownsampleThreshold,
                          &Open3DScene::SetDownsampleThreshold,
                          "Minimum number of points before downsampled point "
                          "clouds are created and used when rendering speed "
                          "is important");
}

}  // namespace rendering
}  // namespace visualization
}  // namespace open3d
