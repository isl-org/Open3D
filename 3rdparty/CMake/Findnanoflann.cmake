# Find nanoflann include dir.
# Once done this will define
#
# nanoflann_FOUND           - true if nanoflann has been found
# nanoflann_INCLUDE_DIR     - where the nanoflann.hpp can be found

if( NOT nanoflann_INCLUDE_DIR )

    find_path( nanoflann_INCLUDE_DIR nanoflann.hpp
               PATHS ${PROJECT_SOURCE_DIR}/3rdparty/nanoflann/include
               NO_DEFAULT_PATH )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(nanoflann DEFAULT_MSG  nanoflann_INCLUDE_DIR )

