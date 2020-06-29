#include "open3d/pipelines/mesh_sampling/TriangleMeshSampling.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

TEST(TriangleMeshSampling, SamplePointsUniformly) {
    auto mesh_empty = geometry::TriangleMesh();
    EXPECT_THROW(
            pipelines::mesh_sampling::SamplePointsUniformly(mesh_empty, 100),
            std::runtime_error);

    std::vector<Eigen::Vector3d> vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    std::vector<Eigen::Vector3i> triangles = {{0, 1, 2}};

    auto mesh_simple = geometry::TriangleMesh();
    mesh_simple.vertices_ = vertices;
    mesh_simple.triangles_ = triangles;

    size_t n_points = 100;
    auto pcd_simple = pipelines::mesh_sampling::SamplePointsUniformly(
            mesh_simple, n_points);
    EXPECT_TRUE(pcd_simple->points_.size() == n_points);
    EXPECT_TRUE(pcd_simple->colors_.size() == 0);
    EXPECT_TRUE(pcd_simple->normals_.size() == 0);

    std::vector<Eigen::Vector3d> colors = {{1, 0, 0}, {1, 0, 0}, {1, 0, 0}};
    std::vector<Eigen::Vector3d> normals = {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}};
    mesh_simple.vertex_colors_ = colors;
    mesh_simple.vertex_normals_ = normals;
    pcd_simple = pipelines::mesh_sampling::SamplePointsUniformly(mesh_simple,
                                                                 n_points);
    EXPECT_TRUE(pcd_simple->points_.size() == n_points);
    EXPECT_TRUE(pcd_simple->colors_.size() == n_points);
    EXPECT_TRUE(pcd_simple->normals_.size() == n_points);

    for (size_t pidx = 0; pidx < n_points; ++pidx) {
        ExpectEQ(pcd_simple->colors_[pidx], Eigen::Vector3d(1, 0, 0));
        ExpectEQ(pcd_simple->normals_[pidx], Eigen::Vector3d(0, 1, 0));
    }

    // use triangle normal instead of the vertex normals
    EXPECT_FALSE(mesh_simple.HasTriangleNormals());
    mesh_simple.ComputeTriangleNormals();
    pcd_simple = pipelines::mesh_sampling::SamplePointsUniformly(
            mesh_simple, n_points, true);
    // the mesh now has triangle normals as a side effect.
    EXPECT_TRUE(mesh_simple.HasTriangleNormals());
    EXPECT_TRUE(pcd_simple->points_.size() == n_points);
    EXPECT_TRUE(pcd_simple->colors_.size() == n_points);
    EXPECT_TRUE(pcd_simple->normals_.size() == n_points);

    for (size_t pidx = 0; pidx < n_points; ++pidx) {
        ExpectEQ(pcd_simple->colors_[pidx], Eigen::Vector3d(1, 0, 0));
        ExpectEQ(pcd_simple->normals_[pidx], Eigen::Vector3d(0, 0, 1));
    }

    // use triangle normal, this time the mesh has no vertex normals
    mesh_simple.vertex_normals_.clear();
    pcd_simple = pipelines::mesh_sampling::SamplePointsUniformly(
            mesh_simple, n_points, true);
    EXPECT_TRUE(pcd_simple->points_.size() == n_points);
    EXPECT_TRUE(pcd_simple->colors_.size() == n_points);
    EXPECT_TRUE(pcd_simple->normals_.size() == n_points);

    for (size_t pidx = 0; pidx < n_points; ++pidx) {
        ExpectEQ(pcd_simple->colors_[pidx], Eigen::Vector3d(1, 0, 0));
        ExpectEQ(pcd_simple->normals_[pidx], Eigen::Vector3d(0, 0, 1));
    }
}

}  // namespace tests
}  // namespace open3d
