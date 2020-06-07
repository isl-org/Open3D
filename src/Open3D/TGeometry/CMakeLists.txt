# Create object library
set(ALL_SOURCE_FILES
    PointCloud.cpp
    PointCloudFactory.cpp
    Geometry3D.cpp
)

add_library(TGeometry OBJECT ${ALL_SOURCE_FILES})
open3d_show_and_abort_on_warning(TGeometry)
open3d_set_global_properties(TGeometry)
open3d_link_3rdparty_libraries(TGeometry)
