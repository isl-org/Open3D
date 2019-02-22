# build
file(GLOB_RECURSE CORE_ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Camera OBJECT ${CORE_ALL_SOURCE_FILES})
