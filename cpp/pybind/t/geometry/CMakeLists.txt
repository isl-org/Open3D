target_sources(pybind PRIVATE
    geometry.cpp
    image.cpp
    pointcloud.cpp
    tensormap.cpp
    trianglemesh.cpp
    tsdf_voxelgrid.cpp
)
# RaycastingScene is not supported on ARM
if(NOT LINUX_AARCH64)
    target_sources(pybind PRIVATE
        raycasting_scene.cpp
    )
endif()