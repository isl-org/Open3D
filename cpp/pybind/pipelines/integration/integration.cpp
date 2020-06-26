// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
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

#include "open3d/geometry/VoxelGrid.h"
#include "open3d/pipelines/integration/ScalableTSDFVolume.h"
#include "open3d/pipelines/integration/TSDFVolume.h"
#include "open3d/pipelines/integration/UniformTSDFVolume.h"

#include "pybind/docstring.h"
#include "pybind/pipelines/integration/integration.h"

namespace open3d {

template <class TSDFVolumeBase = pipelines::integration::TSDFVolume>
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
    py::enum_<pipelines::integration::TSDFVolumeColorType>
            tsdf_volume_color_type(m, "TSDFVolumeColorType", py::arithmetic());
    tsdf_volume_color_type
            .value("NoColor",
                   pipelines::integration::TSDFVolumeColorType::NoColor)
            .value("RGB8", pipelines::integration::TSDFVolumeColorType::RGB8)
            .value("Gray32",
                   pipelines::integration::TSDFVolumeColorType::Gray32)
            .export_values();
    // Trick to write docs without listing the members in the enum class again.
    tsdf_volume_color_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for TSDFVolumeColorType.";
            }),
            py::none(), py::none(), "");

    // open3d.integration.TSDFVolume
    py::class_<pipelines::integration::TSDFVolume,
               PyTSDFVolume<pipelines::integration::TSDFVolume>>
            tsdfvolume(m, "TSDFVolume", R"(Base class of the Truncated
Signed Distance Function (TSDF) volume This volume is usually used to integrate
surface data (e.g., a series of RGB-D images) into a Mesh or PointCloud. The
basic technique is presented in the following paper:

A volumetric method for building complex models from range images

B. Curless and M. Levoy

In SIGGRAPH, 1996)");
    tsdfvolume
            .def("reset", &pipelines::integration::TSDFVolume::Reset,
                 "Function to reset the pipelines::integration::TSDFVolume")
            .def("integrate", &pipelines::integration::TSDFVolume::Integrate,
                 "Function to integrate an RGB-D image into the volume",
                 "image"_a, "intrinsic"_a, "extrinsic"_a)
            .def("extract_point_cloud",
                 &pipelines::integration::TSDFVolume::ExtractPointCloud,
                 "Function to extract a point cloud with normals")
            .def("extract_triangle_mesh",
                 &pipelines::integration::TSDFVolume::ExtractTriangleMesh,
                 "Function to extract a triangle mesh")
            .def_readwrite("voxel_length",
                           &pipelines::integration::TSDFVolume::voxel_length_,
                           "float: Length of the voxel in meters.")
            .def_readwrite("sdf_trunc",
                           &pipelines::integration::TSDFVolume::sdf_trunc_,
                           "float: Truncation value for signed distance "
                           "function (SDF).")
            .def_readwrite("color_type",
                           &pipelines::integration::TSDFVolume::color_type_,
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
    py::class_<pipelines::integration::UniformTSDFVolume,
               PyTSDFVolume<pipelines::integration::UniformTSDFVolume>,
               pipelines::integration::TSDFVolume>
            uniform_tsdfvolume(
                    m, "UniformTSDFVolume",
                    "UniformTSDFVolume implements the classic TSDF "
                    "volume with uniform voxel grid (Curless and Levoy 1996).");
    py::detail::bind_copy_functions<pipelines::integration::UniformTSDFVolume>(
            uniform_tsdfvolume);
    uniform_tsdfvolume
            .def(py::init([](double length, int resolution, double sdf_trunc,
                             pipelines::integration::TSDFVolumeColorType
                                     color_type) {
                     return new pipelines::integration::UniformTSDFVolume(
                             length, resolution, sdf_trunc, color_type);
                 }),
                 "length"_a, "resolution"_a, "sdf_trunc"_a, "color_type"_a)
            .def(py::init([](double length, int resolution, double sdf_trunc,
                             pipelines::integration::TSDFVolumeColorType
                                     color_type,
                             Eigen::Vector3d origin) {
                     return new pipelines::integration::UniformTSDFVolume(
                             length, resolution, sdf_trunc, color_type, origin);
                 }),
                 "length"_a, "resolution"_a, "sdf_trunc"_a, "color_type"_a,
                 "origin"_a)
            .def("__repr__",
                 [](const pipelines::integration::UniformTSDFVolume &vol) {
                     return std::string(
                                    "pipelines::integration::"
                                    "UniformTSDFVolume ") +
                            (vol.color_type_ == pipelines::integration::
                                                        TSDFVolumeColorType::
                                                                NoColor
                                     ? std::string("without color.")
                                     : std::string("with color."));
                 })  // todo: extend
            .def("extract_voxel_point_cloud",
                 &pipelines::integration::UniformTSDFVolume::
                         ExtractVoxelPointCloud,
                 "Debug function to extract the voxel data into a point cloud.")
            .def("extract_voxel_grid",
                 &pipelines::integration::UniformTSDFVolume::ExtractVoxelGrid,
                 "Debug function to extract the voxel data VoxelGrid.")
            .def_readwrite("length",
                           &pipelines::integration::UniformTSDFVolume::length_,
                           "Total length, where ``voxel_length = length / "
                           "resolution``.")
            .def_readwrite(
                    "resolution",
                    &pipelines::integration::UniformTSDFVolume::resolution_,
                    "Resolution over the total length, where "
                    "``voxel_length = length / resolution``");
    docstring::ClassMethodDocInject(m, "UniformTSDFVolume",
                                    "extract_voxel_point_cloud");

    // open3d.integration.ScalableTSDFVolume: open3d.integration.TSDFVolume
    py::class_<pipelines::integration::ScalableTSDFVolume,
               PyTSDFVolume<pipelines::integration::ScalableTSDFVolume>,
               pipelines::integration::TSDFVolume>
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
    py::detail::bind_copy_functions<pipelines::integration::ScalableTSDFVolume>(
            scalable_tsdfvolume);
    scalable_tsdfvolume
            .def(py::init([](double voxel_length, double sdf_trunc,
                             pipelines::integration::TSDFVolumeColorType
                                     color_type,
                             int volume_unit_resolution,
                             int depth_sampling_stride) {
                     return new pipelines::integration::ScalableTSDFVolume(
                             voxel_length, sdf_trunc, color_type,
                             volume_unit_resolution, depth_sampling_stride);
                 }),
                 "voxel_length"_a, "sdf_trunc"_a, "color_type"_a,
                 "volume_unit_resolution"_a = 16, "depth_sampling_stride"_a = 4)
            .def("__repr__",
                 [](const pipelines::integration::ScalableTSDFVolume &vol) {
                     return std::string(
                                    "pipelines::integration::"
                                    "ScalableTSDFVolume ") +
                            (vol.color_type_ == pipelines::integration::
                                                        TSDFVolumeColorType::
                                                                NoColor
                                     ? std::string("without color.")
                                     : std::string("with color."));
                 })
            .def("extract_voxel_point_cloud",
                 &pipelines::integration::ScalableTSDFVolume::
                         ExtractVoxelPointCloud,
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

}  // namespace open3d
