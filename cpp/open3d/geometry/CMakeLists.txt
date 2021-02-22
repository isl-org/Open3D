# build
set(ALL_SOURCE_FILES
    BoundingVolume.cpp
    EstimateNormals.cpp
    Geometry3D.cpp
    HalfEdgeTriangleMesh.cpp
    ISSKeypoints.cpp
    Image.cpp
    ImageFactory.cpp
    IntersectionTest.cpp
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

# create object library
add_library(geometry OBJECT ${ALL_SOURCE_FILES})
open3d_show_and_abort_on_warning(geometry)
open3d_set_global_properties(geometry)
open3d_set_open3d_lib_properties(geometry)
open3d_link_3rdparty_libraries(geometry)
