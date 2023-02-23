// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/pipelines/integration/integration.h"

#include "open3d/geometry/VoxelGrid.h"
#include "open3d/pipelines/integration/ScalableTSDFVolume.h"
#include "open3d/pipelines/integration/TSDFVolume.h"
#include "open3d/pipelines/integration/UniformTSDFVolume.h"
#include "pybind/docstring.h"

namespace open3d {
namespace pipelines {
namespace integration {

template <class TSDFVolumeBase = TSDFVolume>
class PyTSDFVolume : public TSDFVolumeBase {
public:
    using TSDFVolumeBase::TSDFVolumeBase;
    void Reset() override { PYBIND11_OVERLOAD_PURE(void, TSDFVolumeBase, ); }
    void Integrate(const geometry::RGBDImage &image,
                   const camera::PinholeCameraIntrinsic &intrinsic,
                   const Eigen::Matrix4d &extrinsic) override {
        PYBIND11_OVERLOAD_PURE(void, TSDFVolumeBase, image, intrinsic,
                               extrinsic);
    }
    std::shared_ptr<geometry::PointCloud> ExtractPointCloud() override {
        PYBIND11_OVERLOAD_PURE(std::shared_ptr<geometry::PointCloud>,
                               TSDFVolumeBase, );
    }
    std::shared_ptr<geometry::TriangleMesh> ExtractTriangleMesh() override {
        PYBIND11_OVERLOAD_PURE(std::shared_ptr<geometry::TriangleMesh>,
                               TSDFVolumeBase, );
    }
};

void pybind_integration_classes(py::module &m) {
    // open3d.integration.TSDFVolumeColorType
    py::enum_<TSDFVolumeColorType> tsdf_volume_color_type(
            m, "TSDFVolumeColorType", py::arithmetic());
    tsdf_volume_color_type.value("NoColor", TSDFVolumeColorType::NoColor)
            .value("RGB8", TSDFVolumeColorType::RGB8)
            .value("Gray32", TSDFVolumeColorType::Gray32)
            .export_values();
    // Trick to write docs without listing the members in the enum class again.
    tsdf_volume_color_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for TSDFVolumeColorType.";
            }),
            py::none(), py::none(), "");

    // open3d.integration.TSDFVolume
    py::class_<TSDFVolume, PyTSDFVolume<TSDFVolume>> tsdfvolume(
            m, "TSDFVolume", R"(Base class of the Truncated
Signed Distance Function (TSDF) volume This volume is usually used to integrate
surface data (e.g., a series of RGB-D images) into a Mesh or PointCloud. The
basic technique is presented in the following paper:

A volumetric method for building complex models from range images

B. Curless and M. Levoy

In SIGGRAPH, 1996)");
    tsdfvolume
            .def("reset", &TSDFVolume::Reset,
                 "Function to reset the TSDFVolume")
            .def("integrate", &TSDFVolume::Integrate,
                 "Function to integrate an RGB-D image into the volume",
                 "image"_a, "intrinsic"_a, "extrinsic"_a)
            .def("extract_point_cloud", &TSDFVolume::ExtractPointCloud,
                 "Function to extract a point cloud with normals")
            .def("extract_triangle_mesh", &TSDFVolume::ExtractTriangleMesh,
                 "Function to extract a triangle mesh")
            .def_readwrite("voxel_length", &TSDFVolume::voxel_length_,
                           "float: Length of the voxel in meters.")
            .def_readwrite("sdf_trunc", &TSDFVolume::sdf_trunc_,
                           "float: Truncation value for signed distance "
                           "function (SDF).")
            .def_readwrite("color_type", &TSDFVolume::color_type_,
                           "integration.TSDFVolumeColorType: Color type of the "
                           "TSDF volume.");
    docstring::ClassMethodDocInject(m, "TSDFVolume", "extract_point_cloud");
    docstring::ClassMethodDocInject(m, "TSDFVolume", "extract_triangle_mesh");
    docstring::ClassMethodDocInject(
            m, "TSDFVolume", "integrate",
            {{"image", "RGBD image."},
             {"intrinsic", "Pinhole camera intrinsic parameters."},
             {"extrinsic", "Extrinsic parameters."}});
    docstring::ClassMethodDocInject(m, "TSDFVolume", "reset");

    // open3d.integration.UniformTSDFVolume: open3d.integration.TSDFVolume
    py::class_<UniformTSDFVolume, PyTSDFVolume<UniformTSDFVolume>, TSDFVolume>
            uniform_tsdfvolume(
                    m, "UniformTSDFVolume",
                    "UniformTSDFVolume implements the classic TSDF "
                    "volume with uniform voxel grid (Curless and Levoy 1996).");
    py::detail::bind_copy_functions<UniformTSDFVolume>(uniform_tsdfvolume);
    uniform_tsdfvolume
            .def(py::init([](double length, int resolution, double sdf_trunc,
                             TSDFVolumeColorType color_type) {
                     return new UniformTSDFVolume(length, resolution, sdf_trunc,
                                                  color_type);
                 }),
                 "length"_a, "resolution"_a, "sdf_trunc"_a, "color_type"_a)
            .def(py::init([](double length, int resolution, double sdf_trunc,
                             TSDFVolumeColorType color_type,
                             Eigen::Vector3d origin) {
                     return new UniformTSDFVolume(length, resolution, sdf_trunc,
                                                  color_type, origin);
                 }),
                 "length"_a, "resolution"_a, "sdf_trunc"_a, "color_type"_a,
                 "origin"_a)
            .def("__repr__",
                 [](const UniformTSDFVolume &vol) {
                     return std::string(
                                    ""
                                    "UniformTSDFVolume ") +
                            (vol.color_type_ == TSDFVolumeColorType::NoColor
                                     ? std::string("without color.")
                                     : std::string("with color."));
                 })  // todo: extend
            .def("extract_voxel_point_cloud",
                 &UniformTSDFVolume::ExtractVoxelPointCloud,
                 "Debug function to extract the voxel data into a point cloud.")
            .def("extract_voxel_grid", &UniformTSDFVolume::ExtractVoxelGrid,
                 "Debug function to extract the voxel data VoxelGrid.")
            .def("extract_volume_tsdf", &UniformTSDFVolume::ExtractVolumeTSDF,
                 "Debug function to extract the volume TSDF data.")
            .def("extract_volume_color", &UniformTSDFVolume::ExtractVolumeColor,
                 "Debug function to extract the volume color data.")
            .def("inject_volume_tsdf", &UniformTSDFVolume::InjectVolumeTSDF,
                 "Debug function to inject the voxel TSDF data.", "tsdf"_a)
            .def("inject_volume_color", &UniformTSDFVolume::InjectVolumeColor,
                 "Debug function to inject the voxel Color data.", "color"_a)
            .def_readwrite("length", &UniformTSDFVolume::length_,
                           "Total length, where ``voxel_length = length / "
                           "resolution``.")
            .def_readwrite("resolution", &UniformTSDFVolume::resolution_,
                           "Resolution over the total length, where "
                           "``voxel_length = length / resolution``");
    docstring::ClassMethodDocInject(m, "UniformTSDFVolume",
                                    "extract_voxel_point_cloud");

    // open3d.integration.ScalableTSDFVolume: open3d.integration.TSDFVolume
    py::class_<ScalableTSDFVolume, PyTSDFVolume<ScalableTSDFVolume>, TSDFVolume>
            scalable_tsdfvolume(m, "ScalableTSDFVolume", R"(The
ScalableTSDFVolume implements a more memory efficient data structure for
volumetric integration.

This implementation is based on the following repository:
https://github.com/qianyizh/ElasticReconstruction/tree/master/Integrate

An observed depth pixel gives two types of information: (a) an approximation
of the nearby surface, and (b) empty space from the camera to the surface.
They induce two core concepts of volumetric integration: weighted average of
a truncated signed distance function (TSDF), and carving. The weighted
average of TSDF is great in addressing the Gaussian noise along surface
normal and producing a smooth surface output. The carving is great in
removing outlier structures like floating noise pixels and bumps along
structure edges.

Ref: Dense Scene Reconstruction with Points of Interest

Q.-Y. Zhou and V. Koltun

In SIGGRAPH, 2013)");
    py::detail::bind_copy_functions<ScalableTSDFVolume>(scalable_tsdfvolume);
    scalable_tsdfvolume
            .def(py::init([](double voxel_length, double sdf_trunc,
                             TSDFVolumeColorType color_type,
                             int volume_unit_resolution,
                             int depth_sampling_stride) {
                     return new ScalableTSDFVolume(
                             voxel_length, sdf_trunc, color_type,
                             volume_unit_resolution, depth_sampling_stride);
                 }),
                 "voxel_length"_a, "sdf_trunc"_a, "color_type"_a,
                 "volume_unit_resolution"_a = 16, "depth_sampling_stride"_a = 4)
            .def("__repr__",
                 [](const ScalableTSDFVolume &vol) {
                     return std::string(
                                    ""
                                    "ScalableTSDFVolume ") +
                            (vol.color_type_ == TSDFVolumeColorType::NoColor
                                     ? std::string("without color.")
                                     : std::string("with color."));
                 })
            .def("extract_voxel_point_cloud",
                 &ScalableTSDFVolume::ExtractVoxelPointCloud,
                 "Debug function to extract the voxel data into a point "
                 "cloud.");
    docstring::ClassMethodDocInject(m, "ScalableTSDFVolume",
                                    "extract_voxel_point_cloud");
}

void pybind_integration_methods(py::module &m) {
    // Currently empty
}

void pybind_integration(py::module &m) {
    py::module m_submodule =
            m.def_submodule("integration", "Integration pipeline.");
    pybind_integration_classes(m_submodule);
    pybind_integration_methods(m_submodule);
}

}  // namespace integration
}  // namespace pipelines
}  // namespace open3d
