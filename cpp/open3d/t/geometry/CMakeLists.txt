# Create object library
set(ALL_SOURCE_FILES
    PointCloud.cpp
    Image.cpp
    TensorMap.cpp
    TriangleMesh.cpp
    TSDFVoxelGrid.cpp
)

add_library(tgeometry OBJECT ${ALL_SOURCE_FILES})
open3d_show_and_abort_on_warning(tgeometry)
open3d_set_global_properties(tgeometry)
open3d_set_open3d_lib_properties(tgeometry)
open3d_link_3rdparty_libraries(tgeometry)


add_executable(TestVoxelGrid TestVoxelGrid.cpp)
target_link_libraries(TestVoxelGrid ${CMAKE_PROJECT_NAME})
