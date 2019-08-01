# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Odometry OBJECT ${ALL_SOURCE_FILES})
ShowAndAbortOnWarning(Odometry)

# Enforce dependencies for ExternalProject_Add
target_link_libraries(Odometry ${3RDPARTY_LIBRARIES})
