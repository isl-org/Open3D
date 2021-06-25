add_subdirectory(kernel)

add_library(tgeometry OBJECT $<TARGET_OBJECTS:tgeometry_kernel>)

target_sources(tgeometry PRIVATE
    Image.cpp
    PointCloud.cpp
    RGBDImage.cpp
    TensorMap.cpp
    TriangleMesh.cpp
    TSDFVoxelGrid.cpp
    RaycastingScene.cpp
)
# RaycastingScene is not supported on ARM
if(NOT LINUX_AARCH64)
    target_sources(tgeometry PRIVATE
        RaycastingScene.cpp
    )
endif()

open3d_show_and_abort_on_warning(tgeometry)
open3d_set_global_properties(tgeometry)
open3d_set_open3d_lib_properties(tgeometry)
open3d_link_3rdparty_libraries(tgeometry)

if(BUILD_CUDA_MODULE)
    target_include_directories(tgeometry SYSTEM PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
