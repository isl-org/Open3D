#include "open3d/pipelines/mesh_filter/TriangleMeshFilter.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

TEST(TriangleMeshFilter, FilterSharpen) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    mesh->vertices_ = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};
    mesh->triangles_ = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 1}};

    mesh = pipelines::mesh_filter::FilterSharpen(*mesh, 1, 1);
    std::vector<Eigen::Vector3d> ref1 = {
            {0, 0, 0}, {4, 0, 0}, {0, 4, 0}, {-4, 0, 0}, {0, -4, 0}};
    ExpectEQ(mesh->vertices_, ref1);

    mesh = pipelines::mesh_filter::FilterSharpen(*mesh, 9, 0.1);
    std::vector<Eigen::Vector3d> ref2 = {{0, 0, 0},
                                         {42.417997, 0, 0},
                                         {0, 42.417997, 0},
                                         {-42.417997, 0, 0},
                                         {0, -42.417997, 0}};
    ExpectEQ(mesh->vertices_, ref2);
}

TEST(TriangleMeshFilter, FilterSmoothSimple) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    mesh->vertices_ = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};
    mesh->triangles_ = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 1}};

    mesh = pipelines::mesh_filter::FilterSmoothSimple(*mesh, 1);
    std::vector<Eigen::Vector3d> ref1 = {{0, 0, 0},
                                         {0.25, 0, 0},
                                         {0, 0.25, 0},
                                         {-0.25, 0, 0},
                                         {0, -0.25, 0}};
    ExpectEQ(mesh->vertices_, ref1, 1e-4);

    mesh = pipelines::mesh_filter::FilterSmoothSimple(*mesh, 3);
    std::vector<Eigen::Vector3d> ref2 = {{0, 0, 0},
                                         {0.003906, 0, 0},
                                         {0, 0.003906, 0},
                                         {-0.003906, 0, 0},
                                         {0, -0.003906, 0}};
    ExpectEQ(mesh->vertices_, ref2, 1e-4);
}

TEST(TriangleMeshFilter, FilterSmoothLaplacian) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    mesh->vertices_ = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};
    mesh->triangles_ = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 1}};

    mesh = pipelines::mesh_filter::FilterSmoothLaplacian(*mesh, 1, 0.5);
    std::vector<Eigen::Vector3d> ref1 = {
            {0, 0, 0}, {0.5, 0, 0}, {0, 0.5, 0}, {-0.5, 0, 0}, {0, -0.5, 0}};
    ExpectEQ(mesh->vertices_, ref1, 1e-3);

    mesh = pipelines::mesh_filter::FilterSmoothLaplacian(*mesh, 10, 0.5);
    std::vector<Eigen::Vector3d> ref2 = {{0, 0, 0},
                                         {0.000488, 0, 0},
                                         {0, 0.000488, 0},
                                         {-0.000488, 0, 0},
                                         {0, -0.000488, 0}};
    ExpectEQ(mesh->vertices_, ref2, 1e-3);
}

TEST(TriangleMeshFilter, FilterSmoothTaubin) {
    auto mesh = std::make_shared<geometry::TriangleMesh>();
    mesh->vertices_ = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}};
    mesh->triangles_ = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 1}};

    mesh = pipelines::mesh_filter::FilterSmoothTaubin(*mesh, 1, 0.5, -0.53);
    std::vector<Eigen::Vector3d> ref1 = {{0, 0, 0},
                                         {0.765, 0, 0},
                                         {0, 0.765, 0},
                                         {-0.765, 0, 0},
                                         {0, -0.765, 0}};
    ExpectEQ(mesh->vertices_, ref1, 1e-4);

    mesh = pipelines::mesh_filter::FilterSmoothTaubin(*mesh, 10, 0.5, -0.53);
    std::vector<Eigen::Vector3d> ref2 = {{0, 0, 0},
                                         {0.052514, 0, 0},
                                         {0, 0.052514, 0},
                                         {-0.052514, 0, 0},
                                         {0, -0.052514, 0}};
    ExpectEQ(mesh->vertices_, ref2, 1e-4);
}

}  // namespace tests
}  // namespace open3d
