# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Registration OBJECT ${ALL_SOURCE_FILES})
ShowAndAbortOnWarning(Registration)

# Enforce dependencies for ExternalProject_Add
target_link_libraries(Registration ${3RDPARTY_LIBRARIES})
