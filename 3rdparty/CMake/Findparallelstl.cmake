# Find parallelstl include dir.
# Once done this will define
#
# parallelstl_FOUND           - true if parallelstl has been found
# parallelstl_INCLUDE_DIR     - where the parallelstl.hpp can be found

if( NOT parallelstl_INCLUDE_DIR )

    find_path( parallelstl_INCLUDE_DIR pstl/execution
               PATHS ${PROJECT_SOURCE_DIR}/3rdparty/parallelstl/include
               NO_DEFAULT_PATH )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(parallelstl DEFAULT_MSG  parallelstl_INCLUDE_DIR )

