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

#include "Python/integration/integration.h"

#include <Open3D/Integration/TSDFVolume.h>
#include <Open3D/Integration/UniformTSDFVolume.h>
#include <Open3D/Integration/ScalableTSDFVolume.h>

using namespace open3d;

template <class TSDFVolumeBase = integration::TSDFVolume>
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
    py::enum_<integration::TSDFVolumeColorType>(m, "TSDFVolumeColorType",
                                                py::arithmetic())
            .value("None", integration::TSDFVolumeColorType::None)
            .value("RGB8", integration::TSDFVolumeColorType::RGB8)
            .value("Gray32", integration::TSDFVolumeColorType::Gray32)
            .export_values();

    py::class_<integration::TSDFVolume, PyTSDFVolume<integration::TSDFVolume>>
            tsdfvolume(m, "TSDFVolume", "TSDFVolume");
    tsdfvolume
            .def("reset", &integration::TSDFVolume::Reset,
                 "Function to reset the integration::TSDFVolume")
            .def("integrate", &integration::TSDFVolume::Integrate,
                 "Function to integrate an RGB-D image into the volume",
                 "image"_a, "intrinsic"_a, "extrinsic"_a)
            .def("extract_point_cloud",
                 &integration::TSDFVolume::ExtractPointCloud,
                 "Function to extract a point cloud with normals")
            .def("extract_triangle_mesh",
                 &integration::TSDFVolume::ExtractTriangleMesh,
                 "Function to extract a triangle mesh")
            .def_readwrite("voxel_length",
                           &integration::TSDFVolume::voxel_length_)
            .def_readwrite("sdf_trunc", &integration::TSDFVolume::sdf_trunc_)
            .def_readwrite("color_type", &integration::TSDFVolume::color_type_);

    py::class_<integration::UniformTSDFVolume,
               PyTSDFVolume<integration::UniformTSDFVolume>,
               integration::TSDFVolume>
            uniform_tsdfvolume(m, "UniformTSDFVolume", "UniformTSDFVolume");
    py::detail::bind_copy_functions<integration::UniformTSDFVolume>(
            uniform_tsdfvolume);
    uniform_tsdfvolume
            .def(py::init([](double length, int resolution, double sdf_trunc,
                             integration::TSDFVolumeColorType color_type) {
                     return new integration::UniformTSDFVolume(
                             length, resolution, sdf_trunc, color_type);
                 }),
                 "length"_a, "resolution"_a, "sdf_trunc"_a, "color_type"_a)
            .def("__repr__",
                 [](const integration::UniformTSDFVolume &vol) {
                     return std::string("integration::UniformTSDFVolume ") +
                            (vol.color_type_ ==
                                             integration::TSDFVolumeColorType::
                                                     None
                                     ? std::string("without color.")
                                     : std::string("with color."));
                 })  // todo: extend
            .def("extract_voxel_point_cloud",
                 &integration::UniformTSDFVolume::ExtractVoxelPointCloud)
            .def_readwrite("length", &integration::UniformTSDFVolume::length_)
            .def_readwrite("resolution",
                           &integration::UniformTSDFVolume::resolution_);

    py::class_<integration::ScalableTSDFVolume,
               PyTSDFVolume<integration::ScalableTSDFVolume>,
               integration::TSDFVolume>
            scalable_tsdfvolume(m, "ScalableTSDFVolume", "ScalableTSDFVolume");
    py::detail::bind_copy_functions<integration::ScalableTSDFVolume>(
            scalable_tsdfvolume);
    scalable_tsdfvolume
            .def(py::init([](double voxel_length, double sdf_trunc,
                             integration::TSDFVolumeColorType color_type,
                             int volume_unit_resolution,
                             int depth_sampling_stride) {
                     return new integration::ScalableTSDFVolume(
                             voxel_length, sdf_trunc, color_type,
                             volume_unit_resolution, depth_sampling_stride);
                 }),
                 "voxel_length"_a, "sdf_trunc"_a, "color_type"_a,
                 "volume_unit_resolution"_a = 16, "depth_sampling_stride"_a = 4)
            .def("__repr__",
                 [](const integration::ScalableTSDFVolume &vol) {
                     return std::string("integration::ScalableTSDFVolume ") +
                            (vol.color_type_ ==
                                             integration::TSDFVolumeColorType::
                                                     None
                                     ? std::string("without color.")
                                     : std::string("with color."));
                 })
            .def("extract_voxel_point_cloud",
                 &integration::ScalableTSDFVolume::ExtractVoxelPointCloud);
}

void pybind_integration_methods(py::module &m) {
    // Currently empty
}

void pybind_integration(py::module &m) {
    py::module m_submodule = m.def_submodule("integration");
    pybind_integration_classes(m_submodule);
    pybind_integration_methods(m_submodule);
}
