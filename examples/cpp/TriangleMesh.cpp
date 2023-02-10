// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <iostream>

#include "open3d/Open3D.h"

void PaintMesh(open3d::geometry::TriangleMesh &mesh,
               const Eigen::Vector3d &color) {
    mesh.vertex_colors_.resize(mesh.vertices_.size());
    for (size_t i = 0; i < mesh.vertices_.size(); i++) {
        mesh.vertex_colors_[i] = color;
    }
}

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > TriangleMesh sphere");
    utility::LogInfo("    > TriangleMesh merge [file1] [file2]");
    utility::LogInfo("    > TriangleMesh normal [file1] [file2]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace open3d;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    if (argc <= 1 ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    std::string option(argv[1]);
    if (option == "sphere") {
        auto mesh = geometry::TriangleMesh::CreateSphere(0.05);
        mesh->ComputeVertexNormals();
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("sphere.ply", *mesh, true, true);
    } else if (option == "cylinder") {
        auto mesh = geometry::TriangleMesh::CreateCylinder(0.5, 2.0);
        mesh->ComputeVertexNormals();
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("cylinder.ply", *mesh, true, true);
    } else if (option == "cone") {
        auto mesh = geometry::TriangleMesh::CreateCone(0.5, 2.0, 20, 3);
        mesh->ComputeVertexNormals();
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("cone.ply", *mesh, true, true);
    } else if (option == "arrow") {
        auto mesh = geometry::TriangleMesh::CreateArrow();
        mesh->ComputeVertexNormals();
        visualization::DrawGeometries({mesh});
        io::WriteTriangleMesh("arrow.ply", *mesh, true, true);
    } else if (option == "frame") {
        if (argc < 3) {
            auto mesh = geometry::TriangleMesh::CreateCoordinateFrame();
            visualization::DrawGeometries({mesh});
            io::WriteTriangleMesh("frame.ply", *mesh, true, true);
        } else {
            auto mesh = io::CreateMeshFromFile(argv[2]);
            mesh->ComputeVertexNormals();
            auto boundingbox = mesh->GetAxisAlignedBoundingBox();
            auto mesh_frame = geometry::TriangleMesh::CreateCoordinateFrame(
                    boundingbox.GetMaxExtent() * 0.2, boundingbox.min_bound_);
            visualization::DrawGeometries({mesh, mesh_frame});
        }
    } else if (option == "merge") {
        auto mesh1 = io::CreateMeshFromFile(argv[2]);
        auto mesh2 = io::CreateMeshFromFile(argv[3]);
        utility::LogInfo("Mesh1 has {:d} vertices, {:d} triangles.",
                         mesh1->vertices_.size(), mesh1->triangles_.size());
        utility::LogInfo("Mesh2 has {:d} vertices, {:d} triangles.",
                         mesh2->vertices_.size(), mesh2->triangles_.size());
        *mesh1 += *mesh2;
        utility::LogInfo(
                "After merge, Mesh1 has {:d} vertices, {:d} triangles.",
                mesh1->vertices_.size(), mesh1->triangles_.size());
        mesh1->RemoveDuplicatedVertices();
        mesh1->RemoveDuplicatedTriangles();
        mesh1->RemoveDegenerateTriangles();
        mesh1->RemoveUnreferencedVertices();
        utility::LogInfo(
                "After purge vertices, Mesh1 has {:d} vertices, {:d} "
                "triangles.",
                mesh1->vertices_.size(), mesh1->triangles_.size());
        visualization::DrawGeometries({mesh1});
        io::WriteTriangleMesh("temp.ply", *mesh1, true, true);
    } else if (option == "normal") {
        auto mesh = io::CreateMeshFromFile(argv[2]);
        mesh->ComputeVertexNormals();
        io::WriteTriangleMesh(argv[3], *mesh, true, true);
    } else if (option == "scale") {
        auto mesh = io::CreateMeshFromFile(argv[2]);
        double scale = std::stod(argv[4]);
        Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
        trans(0, 0) = trans(1, 1) = trans(2, 2) = scale;
        mesh->Transform(trans);
        io::WriteTriangleMesh(argv[3], *mesh);
    } else if (option == "unify") {
        // unify into (0, 0, 0) - (scale, scale, scale) box
        auto mesh = io::CreateMeshFromFile(argv[2]);
        auto bbox = mesh->GetAxisAlignedBoundingBox();
        double scale1 = std::stod(argv[4]);
        double scale2 = std::stod(argv[5]);
        Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
        trans(0, 0) = trans(1, 1) = trans(2, 2) = scale1 / bbox.GetMaxExtent();
        mesh->Transform(trans);
        trans.setIdentity();
        trans.block<3, 1>(0, 3) =
                Eigen::Vector3d(scale2 / 2.0, scale2 / 2.0, scale2 / 2.0) -
                bbox.GetCenter() * scale1 / bbox.GetMaxExtent();
        mesh->Transform(trans);
        io::WriteTriangleMesh(argv[3], *mesh);
    } else if (option == "distance") {
        auto mesh1 = io::CreateMeshFromFile(argv[2]);
        auto mesh2 = io::CreateMeshFromFile(argv[3]);
        double scale = std::stod(argv[4]);
        mesh1->vertex_colors_.resize(mesh1->vertices_.size());
        geometry::KDTreeFlann kdtree;
        kdtree.SetGeometry(*mesh2);
        std::vector<int> indices(1);
        std::vector<double> dists(1);
        double r = 0.0;
        for (size_t i = 0; i < mesh1->vertices_.size(); i++) {
            kdtree.SearchKNN(mesh1->vertices_[i], 1, indices, dists);
            double color = std::min(sqrt(dists[0]) / scale, 1.0);
            mesh1->vertex_colors_[i] = Eigen::Vector3d(color, color, color);
            r += sqrt(dists[0]);
        }
        utility::LogInfo("Average distance is {:.6f}.",
                         r / (double)mesh1->vertices_.size());
        if (argc > 5) {
            io::WriteTriangleMesh(argv[5], *mesh1);
        }
        visualization::DrawGeometries({mesh1});
    } else if (option == "showboth") {
        auto mesh1 = io::CreateMeshFromFile(argv[2]);
        PaintMesh(*mesh1, Eigen::Vector3d(1.0, 0.75, 0.0));
        auto mesh2 = io::CreateMeshFromFile(argv[3]);
        PaintMesh(*mesh2, Eigen::Vector3d(0.25, 0.25, 1.0));
        std::vector<std::shared_ptr<const geometry::Geometry>> meshes;
        meshes.push_back(mesh1);
        meshes.push_back(mesh2);
        visualization::DrawGeometries(meshes);
    } else if (option == "colormapping") {
        auto mesh = io::CreateMeshFromFile(argv[2]);
        mesh->ComputeVertexNormals();
        camera::PinholeCameraTrajectory trajectory;
        io::ReadIJsonConvertible(argv[3], trajectory);
        if (!utility::filesystem::DirectoryExists("image")) {
            utility::LogWarning("No image!");
            return 0;
        }
        int idx = 3000;
        std::vector<std::shared_ptr<const geometry::Geometry>> ptrs;
        ptrs.push_back(mesh);
        auto mesh_sphere = geometry::TriangleMesh::CreateSphere(0.05);
        Eigen::Matrix4d trans;
        trans.setIdentity();
        trans.block<3, 1>(0, 3) = mesh->vertices_[idx];
        mesh_sphere->Transform(trans);
        mesh_sphere->ComputeVertexNormals();
        ptrs.push_back(mesh_sphere);
        visualization::DrawGeometries(ptrs);

        for (size_t i = 0; i < trajectory.parameters_.size(); i += 10) {
            std::string buffer =
                    fmt::format("image/image_{:06d}.png", (int)i + 1);
            auto image = io::CreateImageFromFile(buffer);
            auto fimage = image->CreateFloatImage();
            Eigen::Vector4d pt_in_camera =
                    trajectory.parameters_[i].extrinsic_ *
                    Eigen::Vector4d(mesh->vertices_[idx](0),
                                    mesh->vertices_[idx](1),
                                    mesh->vertices_[idx](2), 1.0);
            Eigen::Vector3d pt_in_plane =
                    trajectory.parameters_[i].intrinsic_.intrinsic_matrix_ *
                    pt_in_camera.block<3, 1>(0, 0);
            Eigen::Vector3d uv = pt_in_plane / pt_in_plane(2);
            std::cout << pt_in_camera << std::endl;
            std::cout << pt_in_plane << std::endl;
            std::cout << pt_in_plane / pt_in_plane(2) << std::endl;
            auto result = fimage->FloatValueAt(uv(0), uv(1));
            if (result.first) {
                utility::LogInfo("{:.6f}", result.second);
            }
            visualization::DrawGeometries({fimage}, "Test", 1920, 1080);
        }
    }
    return 0;
}
