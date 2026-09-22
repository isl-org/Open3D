target_sources(pybind PRIVATE
    geometry.cpp
    drawablegeometry.cpp
    image.cpp
    lineset.cpp
    pointcloud.cpp
    boundingvolume.cpp
    tensormap.cpp
    trianglemesh.cpp
    voxel_block_grid.cpp
)

if(NOT OPEN3D_DISABLE_EMBREE)
    target_sources(pybind PRIVATE
        raycasting_scene.cpp
    )
endif()
