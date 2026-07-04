add_subdirectory(kernel)

set(OPEN3D_TGEOMETRY_SYCL_SOURCES
    RaycastingScene.cpp
)

open3d_ispc_add_library(tgeometry OBJECT)

target_sources(tgeometry PRIVATE
    Image.cpp
    LineSet.cpp
    BoundingVolume.cpp
    PointCloud.cpp
    RGBDImage.cpp
    TensorMap.cpp
    TriangleMesh.cpp
    TriangleMeshFactory.cpp
    VoxelBlockGrid.cpp
    VtkUtils.cpp
)

if (BUILD_SYCL_MODULE)
    open3d_add_split_module_sources(
        HOST_TARGET tgeometry
        DEVICE_OBJECT tgeometry_xpu
        DEVICE_KIND SYCL
        MODULE_FLAG BUILD_SYCL_MODULE
        SOURCES ${OPEN3D_TGEOMETRY_SYCL_SOURCES}
    )
else()
    target_sources(tgeometry PRIVATE ${OPEN3D_TGEOMETRY_SYCL_SOURCES})
endif()

open3d_show_and_abort_on_warning(tgeometry)
open3d_set_global_properties(tgeometry)
open3d_set_open3d_lib_properties(tgeometry)
open3d_link_3rdparty_libraries(tgeometry)
