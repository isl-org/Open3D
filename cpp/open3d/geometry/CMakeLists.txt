open3d_ispc_add_library(geometry OBJECT)

target_sources(geometry PRIVATE
    BoundingVolume.cpp
    EstimateNormals.cpp
    Geometry3D.cpp
    HalfEdgeTriangleMesh.cpp
    Image.cpp
    ImageFactory.cpp
    IntersectionTest.cpp
    ISSKeypoints.cpp
    KDTreeFlann.cpp
    Line3D.cpp
    LineSet.cpp
    LineSetFactory.cpp
    MeshBase.cpp
    Octree.cpp
    PointCloud.cpp
    PointCloudCluster.cpp
    PointCloudFactory.cpp
    PointCloudSegmentation.cpp
    Qhull.cpp
    RGBDImage.cpp
    RGBDImageFactory.cpp
    SurfaceReconstructionAlphaShape.cpp
    SurfaceReconstructionBallPivoting.cpp
    SurfaceReconstructionPoisson.cpp
    TetraMesh.cpp
    TetraMeshFactory.cpp
    TriangleMesh.cpp
    TriangleMeshDeformation.cpp
    TriangleMeshFactory.cpp
    TriangleMeshSimplification.cpp
    TriangleMeshSubdivide.cpp
    VoxelGrid.cpp
    VoxelGridFactory.cpp
)

open3d_show_and_abort_on_warning(geometry)
open3d_set_global_properties(geometry)
open3d_set_open3d_lib_properties(geometry)
open3d_link_3rdparty_libraries(geometry)
