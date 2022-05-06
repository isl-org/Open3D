open3d_ispc_add_library(tgeometry_kernel OBJECT)

target_sources(tgeometry_kernel PRIVATE
    Image.cpp
    ImageCPU.cpp
    PointCloud.cpp
    PointCloudCPU.cpp
    Transform.cpp
    TransformCPU.cpp
    VoxelBlockGrid.cpp
    VoxelBlockGridCPU.cpp
    VtkUtils.cpp
)

if (BUILD_CUDA_MODULE)
    target_sources(tgeometry_kernel PRIVATE
        ImageCUDA.cu
        NPPImage.cpp
        PointCloudCUDA.cu
        TransformCUDA.cu
        VoxelBlockGridCUDA.cu
    )
endif()

if (WITH_IPPICV)
    target_sources(tgeometry_kernel PRIVATE
        IPPImage.cpp
    )
endif()

open3d_show_and_abort_on_warning(tgeometry_kernel)
open3d_set_global_properties(tgeometry_kernel)
open3d_set_open3d_lib_properties(tgeometry_kernel HIDDEN)
open3d_link_3rdparty_libraries(tgeometry_kernel)

if(BUILD_CUDA_MODULE)
  target_include_directories(tgeometry_kernel SYSTEM
                             PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
