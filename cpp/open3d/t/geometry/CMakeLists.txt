add_library(tgeometry OBJECT)

target_sources(tgeometry PRIVATE
    Image.cpp
    PointCloud.cpp
    RGBDImage.cpp
    TensorMap.cpp
    TriangleMesh.cpp
    TSDFVoxelGrid.cpp
)

target_sources(tgeometry PRIVATE
    kernel/Image.cpp
    kernel/ImageCPU.cpp
    kernel/PointCloud.cpp
    kernel/PointCloudCPU.cpp
    kernel/TSDFVoxelGrid.cpp
    kernel/TSDFVoxelGridCPU.cpp
)

if (BUILD_CUDA_MODULE)
    target_sources(tgeometry PRIVATE
        kernel/ImageCUDA.cu
        kernel/NPPImage.cpp
        kernel/PointCloudCUDA.cu
        kernel/TSDFVoxelGridCUDA.cu
    )
endif()

if (WITH_IPPICV)
    target_sources(tgeometry PRIVATE
        kernel/IPPImage.cpp
    )
endif()

open3d_show_and_abort_on_warning(tgeometry)
open3d_set_global_properties(tgeometry)
open3d_set_open3d_lib_properties(tgeometry)
open3d_link_3rdparty_libraries(tgeometry)

if(BUILD_CUDA_MODULE)
    target_include_directories(tgeometry SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
