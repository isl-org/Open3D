# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

if (WIN32)
    # SurfaceReconstructionPoisson.cpp requires this
    set(CMAKE_C_FLAGS ${CMAKE_CFLAGS} "/bigobj")
    set(CMAKE_CXX_FLAGS ${CMAKE_CFLAGS} "/bigobj")
endif()

# create object library
add_library(Geometry OBJECT ${ALL_SOURCE_FILES})
ShowAndAbortOnWarning(Geometry)

# Enforce 3rd party dependencies
add_dependencies(Geometry build_all_3rd_party_libs)
