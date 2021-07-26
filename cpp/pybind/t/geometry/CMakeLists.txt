target_sources(pybind PRIVATE
    geometry.cpp
    image.cpp
    pointcloud.cpp
    tensormap.cpp
    trianglemesh.cpp
    tsdf_voxelgrid.cpp
)

if (TARGET Open3D::3rdparty_embree)
    target_sources(tgeometry PRIVATE raycasting_scene.cpp)
endif()
