# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Registration OBJECT ${ALL_SOURCE_FILES})
ShowAndAbortOnWarning(Registration)

# Enforce 3rd party dependencies
add_dependencies(Registration build_all_3rd_party_libs)
