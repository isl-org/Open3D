# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Geometry OBJECT ${ALL_SOURCE_FILES})
ShowAndAbortOnWarning(Geometry)

# Enforce 3rd party dependencies
add_dependencies(Geometry build_all_3rd_party_libs)
