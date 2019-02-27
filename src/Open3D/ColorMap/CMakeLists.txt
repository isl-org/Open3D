# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(ColorMap OBJECT ${ALL_SOURCE_FILES})
