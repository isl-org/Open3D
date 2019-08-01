# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Camera OBJECT ${ALL_SOURCE_FILES})
ShowAndAbortOnWarning(Camera)

# Enforce dependencies for ExternalProject_Add
target_link_libraries(Camera ${3RDPARTY_LIBRARIES})
