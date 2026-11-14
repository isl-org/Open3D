add_subdirectory(kernel)

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
)

if(NOT OPEN3D_DISABLE_VTK)
    target_sources(tgeometry PRIVATE
        VtkUtils.cpp
    )
endif()

if(NOT OPEN3D_DISABLE_EMBREE)
    if (BUILD_SYCL_MODULE)
        open3d_sycl_target_sources(tgeometry PRIVATE
            RaycastingScene.cpp
        )
    else()
        target_sources(tgeometry PRIVATE
            RaycastingScene.cpp
        )
    endif()
else()
    # Embree is disabled (e.g. Windows ARM64): compile stub implementation
    # so that symbols referenced by TriangleMesh.cpp are still available.
    target_sources(tgeometry PRIVATE
        RaycastingScene_stub.cpp
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
