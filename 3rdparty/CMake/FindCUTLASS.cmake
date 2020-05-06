# Find CUTLASS include dir.
# Once done this will define
#
# CUTLASS_FOUND           - true if CUTLASS has been found
# CUTLASS_INCLUDE_DIR     - where the CUTLASS.hpp can be found

if( NOT CUTLASS_INCLUDE_DIR )

    find_path( CUTLASS_INCLUDE_DIR cutlass/gemm/gemm.h
               PATHS ${PROJECT_SOURCE_DIR}/3rdparty/cutlass
               NO_DEFAULT_PATH )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUTLASS DEFAULT_MSG  CUTLASS_INCLUDE_DIR )

