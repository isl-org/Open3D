# open3d_sycl_target_sources(name ...)
#
# When BUILD_SYCL_MODULE=ON, set SYCL-specific compile flags for the listed
# source files and call target_sources(). If BUILD_SYCL_MODULE=OFF, this
# function directly calls target_sources().
#
# Note: this is not a perfect forwarding to target_sources(), as it only support
# limited set of arguments. See the example usage below.
#
# This is different from CUDA. When BUILD_CUDA_MODULE is ON, all source files
# will have CUDA include directories and link with CUDA libraries. However,
# if BUILD_SYCL_MODULE is ON, only the source files that are added with
# open3d_sycl_target_sources() will have SYCL include directories and link with
# SYCL libraries. One advantage of this is to speed up recompilation, as ccache
# cannot cache SYCL compileations for now.
#
# Example usage:
#   open3d_sycl_target_sources(core PRIVATE a.cpp b.cpp)
#   open3d_sycl_target_sources(core PUBLIC a.cpp b.cpp)
#   open3d_sycl_target_sources(core VERBOSE PRIVATE a.cpp b.cpp)
#   open3d_sycl_target_sources(core VERBOSE PUBLIC a.cpp b.cpp)
function(open3d_sycl_target_sources target)
    cmake_parse_arguments(arg "PUBLIC;PRIVATE;INTERFACE;VERBOSE" "" "" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        if(arg_PUBLIC)
            target_sources(${target} PUBLIC ${arg_UNPARSED_ARGUMENTS})
            message(STATUS "open3d_sycl_target_sources(${target}): PUBLIC")
        elseif (arg_PRIVATE)
            target_sources(${target} PRIVATE ${arg_UNPARSED_ARGUMENTS})
            message(STATUS "open3d_sycl_target_sources(${target}): PRIVATE")
        elseif (arg_INTERFACE)
            target_sources(${target} INTERFACE ${arg_UNPARSED_ARGUMENTS})
            message(STATUS "open3d_sycl_target_sources(${target}): INTERFACE")
        else()
            message(FATAL_ERROR "Invalid syntax: open3d_sycl_target_sources(${name} ${ARGN})")
        endif()

        if(BUILD_SYCL_MODULE)
            foreach(sycl_file IN LISTS arg_UNPARSED_ARGUMENTS)
                # COMPILE_OPTIONS is a list, COMPILE_FLAGS is a string.
                # https://stackoverflow.com/a/24303754/1255535
                if(ENABLE_SYCL_AOT_ADL)
                    set_source_files_properties(${sycl_file} PROPERTIES
                        COMPILE_FLAGS "-fsycl -fsycl-unnamed-lambda -fsycl-targets=spir64_gen"
                    )
                else()
                    set_source_files_properties(${sycl_file} PROPERTIES
                        COMPILE_FLAGS "-fsycl -fsycl-unnamed-lambda"
                    )
                endif()

                # __OPEN3D_SYCLCC__ is analogous to __CUDACC__.
                # BUILD_SYCL_MODULE is analogous to BUILD_CUDA_MODULE.
                set_source_files_properties(${sycl_file} PROPERTIES
                    COMPILE_DEFINITIONS __OPEN3D_SYCLCC__=1
                )
                if(arg_VERBOSE)
                    message(STATUS "open3d_sycl_target_sources(${target}): marked ${sycl_file} as SYCL code")
                endif()
            endforeach()
        endif()
    endif()
endfunction()
