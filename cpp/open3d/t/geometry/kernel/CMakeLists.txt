set(OPEN3D_TGEOMETRY_KERNEL_CUDA_SOURCES
    ImageCUDA.cu
    NPPImage.cpp
    PointCloudCUDA.cu
    TriangleMeshCUDA.cu
    TransformCUDA.cu
    VoxelBlockGridCUDA.cu
)

open3d_ispc_add_library(tgeometry_kernel OBJECT)
set_target_properties(tgeometry_kernel PROPERTIES CXX_VISIBILITY_PRESET "hidden")

target_sources(tgeometry_kernel PRIVATE
    Image.cpp
    ImageCPU.cpp
    PCAPartition.cpp
    PointCloud.cpp
    PointCloudCPU.cpp
    Metrics.cpp
    MinimumOBB.cpp
    MinimumOBE.cpp
    TriangleMesh.cpp
    TriangleMeshCPU.cpp
    Transform.cpp
    TransformCPU.cpp
    UVUnwrapping.cpp
    VoxelBlockGrid.cpp
    VoxelBlockGridCPU.cpp
    mikktspace.c
)

open3d_add_split_module_sources(
    HOST_TARGET tgeometry_kernel
    DEVICE_OBJECT tgeometry_kernel_cuda
    DEVICE_KIND CUDA
    MODULE_FLAG BUILD_CUDA_MODULE
    HIDDEN
    SOURCES ${OPEN3D_TGEOMETRY_KERNEL_CUDA_SOURCES}
)

if (WITH_IPP)
    target_sources(tgeometry_kernel PRIVATE
        IPPImage.cpp
    )
endif()


open3d_show_and_abort_on_warning(tgeometry_kernel)
open3d_set_global_properties(tgeometry_kernel)
open3d_set_open3d_lib_properties(tgeometry_kernel HIDDEN)
open3d_link_3rdparty_libraries(tgeometry_kernel)

