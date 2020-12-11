#include <fmt/format.h>

#include "open3d/Open3D.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/t/geometry/Image.h"
#include "open3d/t/geometry/PointCloud.h"
#include "open3d/t/geometry/TSDFVoxelGrid.h"
#include "open3d/t/geometry/TriangleMesh.h"
#include "open3d/utility/Console.h"

using namespace open3d;
using namespace open3d::core;

template <class T, int M, int N, int A>
Tensor FromEigen(const Eigen::Matrix<T, M, N, A>& matrix) {
    Dtype dtype = Dtype::FromType<T>();
    Eigen::Matrix<T, M, N, Eigen::RowMajor> matrix_row_major = matrix;
    return Tensor(matrix_row_major.data(), {matrix.rows(), matrix.cols()},
                  dtype);
}

int main(int argc, char** argv) {
    std::string root_path = argv[1];
    // std::shared_ptr<geometry::TriangleMesh> mesh_io =
    //         io::CreateMeshFromFile(root_path + "/scene/integrated.ply");

    // auto mesh = t::geometry::TriangleMesh::FromLegacyTrangleMesh(*mesh_io);
    // auto mesh_legacy = std::make_shared<geometry::TriangleMesh>(
    //         mesh.ToLegacyTriangleMesh());
    // visualization::DrawGeometries({mesh_legacy});

    Tensor intrinsic = Tensor(
            std::vector<float>({525.0, 0, 319.5, 0, 525.0, 239.5, 0, 0, 1}),
            {3, 3}, Dtype::Float32);

    auto trajectory = io::CreatePinholeCameraTrajectoryFromFile(
            fmt::format("{}/trajectory.log", root_path));

    std::vector<Device> devices{Device("CUDA:0"), Device("CPU:0")};

    for (auto device : devices) {
        core::cuda::ReleaseCache();

        t::geometry::TSDFVoxelGrid voxel_grid({{"tsdf", 4}, {"weight", 4}},
                                              3.0 / 512, 0.04, 16, 100, device);

        std::vector<std::shared_ptr<const open3d::geometry::Geometry>>
                geometries;
        for (int i = 0; i < trajectory->parameters_.size(); ++i) {
            // for (int i = 0; i < 100; ++i) {
            /// Load image
            std::string image_path =
                    fmt::format("{}/depth/{:06d}.png", root_path, i + 1);
            std::string color_path =
                    fmt::format("{}/color/{:06d}.png", root_path, i + 1);

            std::shared_ptr<geometry::Image> depth_legacy =
                    io::CreateImageFromFile(image_path);
            std::shared_ptr<geometry::Image> color_legacy =
                    io::CreateImageFromFile(color_path);

            t::geometry::Image depth =
                    t::geometry::Image::FromLegacyImage(*depth_legacy, device);
            t::geometry::Image color =
                    t::geometry::Image::FromLegacyImage(*color_legacy, device);

            Eigen::Matrix4f extrinsic_eigen =
                    trajectory->parameters_[i].extrinsic_.cast<float>();
            Tensor extrinsic = FromEigen(extrinsic_eigen).Copy(device);

            // auto pcd = t::geometry::PointCloud::CreateFromDepthImage(
            //         depth, intrinsic, 1000.0);
            // pcd.Transform(extrinsic.Inverse());
            // auto pcd_down = pcd.VoxelDownSample(16 * 3.0 / 512);

            // auto pcd_legacy =
            // std::make_shared<open3d::geometry::PointCloud>(
            //         pcd_down.ToLegacyPointCloud());
            // geometries.push_back(pcd_legacy);
            // if (i % 10 == 0) {
            //     visualization::DrawGeometries(geometries);
            // }
            utility::Timer timer;
            timer.Start();
            voxel_grid.Integrate(depth, color, intrinsic, extrinsic);
            timer.Stop();
            utility::LogInfo("{}: Integration takes {}", i,
                             timer.GetDuration());
        }

        utility::Timer timer;
        // timer.Start();
        // auto pcd = voxel_grid.ExtractSurfacePoints();
        // timer.Stop();
        // utility::LogInfo("Point Extraction takes {}",
        // timer.GetDuration());

        // timer.Start();
        // auto pcd_legacy =
        // std::make_shared<open3d::geometry::PointCloud>(
        //         pcd.ToLegacyPointCloud());
        // timer.Stop();
        // utility::LogInfo("Conversion takes {}", timer.GetDuration());

        // timer.Start();
        // open3d::io::WritePointCloud("pcd_" + device.ToString() +
        // ".ply",
        //                             *pcd_legacy);
        // timer.Stop();
        // utility::LogInfo("IO takes {}", timer.GetDuration());
        // open3d::visualization::DrawGeometries({pcd_legacy});

        timer.Start();
        auto mesh = voxel_grid.ExtractSurfaceMesh();
        timer.Stop();
        utility::LogInfo("Mesh Extraction takes {}", timer.GetDuration());

        timer.Start();
        auto mesh_legacy = std::make_shared<geometry::TriangleMesh>(
                mesh.ToLegacyTriangleMesh());
        timer.Stop();
        utility::LogInfo("Conversion takes {}", timer.GetDuration());

        timer.Start();
        open3d::io::WriteTriangleMesh("mesh_" + device.ToString() + ".ply",
                                      *mesh_legacy);
        timer.Stop();
        utility::LogInfo("IO takes {}", timer.GetDuration());

        // open3d::visualization::DrawGeometries({mesh_legacy});

        // auto mesh = voxel_grid.ExtractSurfaceMesh();
        // auto mesh_pcd = t::geometry::PointCloud(mesh.GetVertices());
        // // mesh_pcd.SetPointNormals(mesh.GetVertexNormals());
        // auto mesh_pcd_legacy =
        // std::make_shared<open3d::geometry::PointCloud>(
        //         mesh_pcd.ToLegacyPointCloud());
        // // mesh_pcd_legacy->EstimateNormals();
        // open3d::io::WritePointCloud("mesh_pcd_" + device.ToString() +
        // ".ply",
        //                             *mesh_pcd_legacy);
        // open3d::visualization::DrawGeometries({mesh_pcd_legacy});
    }
}
