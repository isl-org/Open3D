# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Utility OBJECT ${ALL_SOURCE_FILES})
ShowAndAbortOnWarning(Utility)

# Enforce 3rd party dependencies
add_dependencies(Utility build_all_3rd_party_libs)
