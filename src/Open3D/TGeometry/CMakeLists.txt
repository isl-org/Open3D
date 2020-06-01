# Create object library
set(ALL_SOURCE_FILES
    PointCloud.cpp
    PointCloudFactory.cpp
    Geometry3D.cpp
)
add_library(TGeometry OBJECT ${ALL_SOURCE_FILES})
ShowAndAbortOnWarning(TGeometry)

# Enforce 3rd party dependencies
add_dependencies(TGeometry build_all_3rd_party_libs)
