# Try to find Intel Thread Building Blocks library and include dir.
# Once done this will define
#
# TBB_FOUND           - true if TBB has been found
# TBB_INCLUDE_DIR     - where the tbb/parallel_for.h can be found
# TBB_LIBRARY         - TBB library
# TBB_MALLOC_LIBRARY  - TBB malloc library


if( NOT TBB_INCLUDE_DIR )
    # try to find the header inside a conda environment first
    find_path( TBB_INCLUDE_DIR tbb/parallel_for.h
               HINTS $ENV{CONDA_PREFIX}/include )
endif()

if( NOT TBB_LIBRARY )
    find_library( TBB_LIBRARY tbb 
            HINTS $ENV{CONDA_PREFIX}/lib )
endif()

if( NOT TBB_MALLOC_LIBRARY )
    find_library( TBB_MALLOC_LIBRARY tbbmalloc
            HINTS $ENV{CONDA_PREFIX}/lib )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB  DEFAULT_MSG  TBB_INCLUDE_DIR TBB_LIBRARY TBB_MALLOC_LIBRARY)

