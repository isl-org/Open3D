#include <fmt/format.h>

#include "open3d/Open3D.h"
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
    std::shared_ptr<geometry::TriangleMesh> mesh_io =
            io::CreateMeshFromFile(root_path + "/scene/integrated.ply");

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
        t::geometry::TSDFVoxelGrid voxel_grid({{"tsdf", 1}, {"weight", 1}},
                                              3.0 / 512, 0.04, 16, 10, device);
        std::vector<std::shared_ptr<const open3d::geometry::Geometry>>
                geometries;
        for (int i = 0; i < 300; ++i) {
            /// Load image
            std::string image_path =
                    fmt::format("{}/depth/{:06d}.png", root_path, i + 1);
            std::shared_ptr<geometry::Image> depth_legacy =
                    io::CreateImageFromFile(image_path);
            t::geometry::Image depth =
                    t::geometry::Image::FromLegacyImage(*depth_legacy, device);

            Eigen::Matrix4f extrinsic_eigen =
                    trajectory->parameters_[i].extrinsic_.cast<float>();
            Tensor extrinsic = FromEigen(extrinsic_eigen).Copy(device);

            // auto pcd = t::geometry::PointCloud::CreateFromDepthImage(
            //         depth, intrinsic, 1000.0);
            // pcd.Transform(extrinsic.Inverse());
            // auto pcd_down = pcd.VoxelDownSample(16 * 3.0 / 512);

            // auto pcd_legacy = std::make_shared<open3d::geometry::PointCloud>(
            //         pcd_down.ToLegacyPointCloud());
            // geometries.push_back(pcd_legacy);
            // if (i % 10 == 0) {
            //     visualization::DrawGeometries(geometries);
            // }
            utility::Timer timer;
            timer.Start();
            voxel_grid.Integrate(depth, intrinsic, extrinsic);
            timer.Stop();
            utility::LogInfo("{}: Integration takes {}", i,
                             timer.GetDuration());
        }

        auto pcd = voxel_grid.ExtractSurfacePoints();
        auto pcd_legacy = std::make_shared<open3d::geometry::PointCloud>(
                pcd.ToLegacyPointCloud());
        open3d::io::WritePointCloud("pcd.ply", *pcd_legacy);

        auto mesh = voxel_grid.ExtractSurfaceMesh();
        auto mesh_legacy = std::make_shared<geometry::TriangleMesh>(
                mesh.ToLegacyTriangleMesh());
        open3d::io::WriteTriangleMesh("mesh.ply", *mesh_legacy);
        mesh_legacy->ComputeVertexNormals();
        open3d::visualization::DrawGeometries({mesh_legacy});
    }
}
