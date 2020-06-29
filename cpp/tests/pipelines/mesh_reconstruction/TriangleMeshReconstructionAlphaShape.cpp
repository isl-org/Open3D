#include "open3d/pipelines/mesh_reconstruction/TriangleMeshReconstruction.h"
#include "tests/UnitTest.h"

namespace open3d {
namespace tests {

TEST(TriangleMeshReconstructionAlphaShape, ReconstructAlphaShape) {
    geometry::PointCloud pcd;
    pcd.points_ = {
            {0.765822, 1.000000, 0.486627}, {0.034963, 1.000000, 0.632086},
            {0.000000, 0.093962, 0.028012}, {0.000000, 0.910057, 0.049732},
            {0.017178, 0.000000, 0.946382}, {0.972485, 0.000000, 0.431460},
            {0.794109, 0.033417, 1.000000}, {0.700868, 0.648112, 1.000000},
            {0.164379, 0.516339, 1.000000}, {0.521248, 0.377170, 0.000000}};
    geometry::TriangleMesh mesh_gt;
    mesh_gt.vertices_ = {
            {0.521248, 0.377170, 0.000000}, {0.017178, 0.000000, 0.946382},
            {0.972485, 0.000000, 0.431460}, {0.000000, 0.093962, 0.028012},
            {0.164379, 0.516339, 1.000000}, {0.700868, 0.648112, 1.000000},
            {0.765822, 1.000000, 0.486627}, {0.794109, 0.033417, 1.000000},
            {0.034963, 1.000000, 0.632086}, {0.000000, 0.910057, 0.049732}};
    mesh_gt.triangles_ = {{0, 2, 3}, {1, 2, 3}, {2, 5, 6}, {0, 2, 6},
                          {4, 5, 7}, {2, 5, 7}, {1, 2, 7}, {1, 4, 7},
                          {1, 4, 8}, {1, 3, 8}, {4, 5, 8}, {5, 6, 8},
                          {3, 8, 9}, {0, 3, 9}, {6, 8, 9}, {0, 6, 9}};

    auto mesh_es =
            pipelines::mesh_reconstruction::ReconstructAlphaShape(pcd, 1);

    double threshold = 1e-6;
    ExpectEQ(mesh_es->vertices_, mesh_gt.vertices_, threshold);
    ExpectEQ(mesh_es->vertex_normals_, mesh_gt.vertex_normals_, threshold);
    ExpectEQ(mesh_es->vertex_colors_, mesh_gt.vertex_colors_, threshold);
    ExpectEQ(mesh_es->triangles_, mesh_gt.triangles_);
    ExpectEQ(mesh_es->triangle_normals_, mesh_gt.triangle_normals_, threshold);
}

}  // namespace tests
}  // namespace open3d
