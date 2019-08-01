# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(ColorMap OBJECT ${ALL_SOURCE_FILES})
ShowAndAbortOnWarning(ColorMap)

# Enforce dependencies for ExternalProject_Add
target_link_libraries(ColorMap ${3RDPARTY_LIBRARIES})
