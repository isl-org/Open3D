# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(ColorMap OBJECT ${ALL_SOURCE_FILES})
ShowAndAbortOnWarning(ColorMap)

# Enforce 3rd party dependencies
add_dependencies(ColorMap build_all_3rd_party_libs)
