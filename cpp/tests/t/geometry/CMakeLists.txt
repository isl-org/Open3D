target_sources(tests PRIVATE
    Image.cpp
    LineSet.cpp
    PointCloud.cpp
    TensorMap.cpp
    TriangleMesh.cpp
    AxisAlignedBoundingBox.cpp
    OrientedBoundingBox.cpp
    OrientedBoundingEllipsoid.cpp
    VoxelBlockGrid.cpp
)

if(NOT OPEN3D_DISABLE_VTK)
    target_sources(tests PRIVATE
        VtkUtils.cpp
    )
endif()
