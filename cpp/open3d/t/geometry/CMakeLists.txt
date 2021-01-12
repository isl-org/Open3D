# Create object library
set(T_GEOMETRY_KERNEL_SRC
    kernel/PointCloud.cpp
    kernel/PointCloudCPU.cpp
    kernel/TSDFVoxelGrid.cpp
    kernel/TSDFVoxelGridCPU.cpp
)

set(T_GEOMETRY_KERNEL_CUDA_SRC
    kernel/PointCloudCUDA.cu
    kernel/TSDFVoxelGridCUDA.cu
)

set(T_GEOMETRY_SRC
    PointCloud.cpp
    Image.cpp
    RGBDImage.cpp
    TensorMap.cpp
    TriangleMesh.cpp
    TSDFVoxelGrid.cpp
)

if (BUILD_CUDA_MODULE)
    set(ALL_T_GEOMETRY_SRC
      ${T_GEOMETRY_KERNEL_SRC}
      ${T_GEOMETRY_KERNEL_CUDA_SRC}
      ${T_GEOMETRY_SRC}
    )
else()
    set(ALL_T_GEOMETRY_SRC
      ${T_GEOMETRY_KERNEL_SRC}
      ${T_GEOMETRY_SRC}
    )
endif()

add_library(tgeometry OBJECT ${ALL_T_GEOMETRY_SRC})
open3d_show_and_abort_on_warning(tgeometry)
open3d_set_global_properties(tgeometry)
open3d_set_open3d_lib_properties(tgeometry)
open3d_link_3rdparty_libraries(tgeometry)

if(BUILD_CUDA_MODULE)
    target_include_directories(tgeometry SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
