# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Registration OBJECT ${ALL_SOURCE_FILES})
