# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Integration OBJECT ${ALL_SOURCE_FILES})
ShowAndAbortOnWarning(Integration)

# Enforce dependencies for ExternalProject_Add
target_link_libraries(Integration ${3RDPARTY_LIBRARIES})
