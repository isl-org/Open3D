# Find CUB include dir.
#
# Once done this will define:
# - CUB_FOUND           - true if CUB has been found
# - CUB_INCLUDE_DIR     - where the CUB.hpp can be found

if(NOT CUB_INCLUDE_DIR)
    find_path(
        CUB_INCLUDE_DIR cub/cub.cuh
        # CUDA 11 and later includes CUB. 
        # Make sure to search in the toolkit include dir first.
        PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR}/3rdparty/cub
        NO_DEFAULT_PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUB DEFAULT_MSG CUB_INCLUDE_DIR)
