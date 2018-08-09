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

#include "open3d_core.h"
#include "open3d_core_trampoline.h"

#include <Core/Integration/TSDFVolume.h>
#include <Core/Integration/UniformTSDFVolume.h>
#include <Core/Integration/ScalableTSDFVolume.h>

using namespace open3d;

template <class TSDFVolumeBase = TSDFVolume>
class PyTSDFVolume : public TSDFVolumeBase
{
public:
    using TSDFVolumeBase::TSDFVolumeBase;
    void Reset() override {
        PYBIND11_OVERLOAD_PURE(void, TSDFVolumeBase, );
    }
    void Integrate(const RGBDImage &image,
            const PinholeCameraIntrinsic &intrinsic,
            const Eigen::Matrix4d &extrinsic) override {
        PYBIND11_OVERLOAD_PURE(void, TSDFVolumeBase, image, intrinsic,
                extrinsic);
    }
    std::shared_ptr<PointCloud> ExtractPointCloud() override {
        PYBIND11_OVERLOAD_PURE(std::shared_ptr<PointCloud>, TSDFVolumeBase, );
    }
    std::shared_ptr<TriangleMesh> ExtractTriangleMesh() override {
        PYBIND11_OVERLOAD_PURE(std::shared_ptr<TriangleMesh>, TSDFVolumeBase, );
    }
};

void pybind_integration(py::module &m)
{
    py::enum_<TSDFVolumeColorType>(m,
        "TSDFVolumeColorType", py::arithmetic())
        .value("None",
        TSDFVolumeColorType::None)
        .value("RGB8",
        TSDFVolumeColorType::RGB8)
        .value("Gray32",
        TSDFVolumeColorType::Gray32)
        .export_values();

    py::class_<TSDFVolume, PyTSDFVolume<TSDFVolume>>
            tsdfvolume(m, "TSDFVolume");
    tsdfvolume
        .def("reset", &TSDFVolume::Reset, "Function to reset the TSDFVolume")
        .def("integrate", &TSDFVolume::Integrate,
                "Function to integrate an RGB-D image into the volume",
                "image"_a, "intrinsic"_a, "extrinsic"_a)
        .def("extract_point_cloud", &TSDFVolume::ExtractPointCloud,
                "Function to extract a point cloud with normals")
        .def("extract_triangle_mesh", &TSDFVolume::ExtractTriangleMesh,
                "Function to extract a triangle mesh")
        .def_readwrite("voxel_length", &TSDFVolume::voxel_length_)
        .def_readwrite("sdf_trunc", &TSDFVolume::sdf_trunc_)
        .def_readwrite("color_type", &TSDFVolume::color_type_);

    py::class_<UniformTSDFVolume, PyTSDFVolume<UniformTSDFVolume>, TSDFVolume>
            uniform_tsdfvolume(m, "UniformTSDFVolume");
    py::detail::bind_copy_functions<UniformTSDFVolume>(
            uniform_tsdfvolume);
    uniform_tsdfvolume
        .def(py::init([](double length, int resolution,
                double sdf_trunc, TSDFVolumeColorType color_type) {
            return new UniformTSDFVolume(
                    length, resolution, sdf_trunc, color_type);
        }), "length"_a, "resolution"_a, "sdf_trunc"_a, "color_type"_a)
        .def("__repr__", [](const UniformTSDFVolume &vol) {
            return std::string("UniformTSDFVolume ") +
                    (vol.color_type_ == TSDFVolumeColorType::None ?
                    std::string("without color.") :
                    std::string("with color.")); }) // todo: extend
        .def("extract_voxel_point_cloud",
                &UniformTSDFVolume::ExtractVoxelPointCloud)
        .def_readwrite("length", &UniformTSDFVolume::length_)
        .def_readwrite("resolution", &UniformTSDFVolume::resolution_);

    py::class_<ScalableTSDFVolume, PyTSDFVolume<ScalableTSDFVolume>, TSDFVolume>
            scalable_tsdfvolume(m, "ScalableTSDFVolume");
    py::detail::bind_copy_functions<ScalableTSDFVolume>(
            scalable_tsdfvolume);
    scalable_tsdfvolume
        .def(py::init([](double voxel_length, double sdf_trunc,
                TSDFVolumeColorType color_type, int volume_unit_resolution,
                int depth_sampling_stride) {
            return new ScalableTSDFVolume(voxel_length, sdf_trunc,
                color_type, volume_unit_resolution, depth_sampling_stride);
        }), "voxel_length"_a, "sdf_trunc"_a, "color_type"_a,
                "volume_unit_resolution"_a = 16, "depth_sampling_stride"_a = 4)
        .def("__repr__", [](const ScalableTSDFVolume &vol) {
            return std::string("ScalableTSDFVolume ") +
                    (vol.color_type_ == TSDFVolumeColorType::None ?
                    std::string("without color.") :
                    std::string("with color."));
    })
        .def("extract_voxel_point_cloud",
                &ScalableTSDFVolume::ExtractVoxelPointCloud);
}

void pybind_integration_methods(py::module &m)
{
}
