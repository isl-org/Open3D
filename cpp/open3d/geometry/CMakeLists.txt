# build
file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(geometry OBJECT ${ALL_SOURCE_FILES})
open3d_show_and_abort_on_warning(geometry)
open3d_set_global_properties(geometry)
open3d_link_3rdparty_libraries(geometry)

if (BUILD_CUDA_MODULE)
    # We still need to explicitly link against CUDA libraries.
    # See: https://stackoverflow.com/a/48540499/1255535.
    # Consider removing dependencies of cuda headers in the future.
    target_link_libraries(geometry PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(geometry SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
