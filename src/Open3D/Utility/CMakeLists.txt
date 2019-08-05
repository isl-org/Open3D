# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Utility OBJECT ${ALL_SOURCE_FILES})
ShowAndAbortOnWarning(Utility)

# Enforce dependencies for ExternalProject_Add
target_link_libraries(Utility ${3RDPARTY_LIBRARIES})
