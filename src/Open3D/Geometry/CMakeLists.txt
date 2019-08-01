# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Geometry OBJECT ${ALL_SOURCE_FILES})
ShowAndAbortOnWarning(Geometry)

# Enforce dependencies for ExternalProject_Add
target_link_libraries(Geometry ${3RDPARTY_LIBRARIES})
