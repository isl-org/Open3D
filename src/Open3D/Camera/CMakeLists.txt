# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Camera OBJECT ${ALL_SOURCE_FILES})
ShowAndAbortOnWarning(Camera)

# Enforce 3rd party dependencies
add_dependencies(Camera build_all_3rd_party_libs)
