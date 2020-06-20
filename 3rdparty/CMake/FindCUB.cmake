# Find CUB include dir.
#
# Once done this will define:
# - CUB_FOUND           - true if CUB has been found
# - CUB_INCLUDE_DIR     - where the CUB.hpp can be found

if(NOT CUB_INCLUDE_DIR)
    find_path(
        CUB_INCLUDE_DIR cub/cub.cuh
        PATHS ${PROJECT_SOURCE_DIR}/3rdparty/cub
        NO_DEFAULT_PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUB DEFAULT_MSG CUB_INCLUDE_DIR)
