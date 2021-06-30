target_sources(pybind PRIVATE
    geometry.cpp
    image.cpp
    pointcloud.cpp
    tensormap.cpp
    trianglemesh.cpp
    tsdf_voxelgrid.cpp
)
target_sources(pybind PRIVATE
    raycasting_scene.cpp
)
