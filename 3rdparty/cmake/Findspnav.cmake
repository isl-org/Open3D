# Try to find Mesa off-screen library and include dir.
# Once done this will define
#
# SPNAV_FOUND        - true if spnav has been found
# SPNAV_INCLUDE_DIR  - where the spnav.h can be found
# SPNAV_LIBRARY      - Link this to use spnav

if(NOT SPNAV_INCLUDE_DIR)

  # If we have a root defined look there first
  if(SPNAV_ROOT)
    find_path(SPNAV_INCLUDE_DIR spnav.h PATHS ${SPNAV_ROOT}/include
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT SPNAV_INCLUDE_DIR)
    find_path(SPNAV_INCLUDE_DIR spnav.h PATHS
      /usr/include
      /usr/local/include
    )
  endif()
endif()

if(NOT SPNAV_LIBRARY)
  # If we have a root defined look there first
  if(SPNAV_ROOT)
    find_library(SPNAV_LIBRARY NAMES spnav PATHS ${SPNAV_ROOT}/lib
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT SPNAV_LIBRARY)
    find_library(SPNAV_LIBRARY NAMES spnav PATHS
      /usr/lib
      /usr/lib/x86_64-linux-gnu
      /usr/local/lib
    )
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(spnav  DEFAULT_MSG  SPNAV_LIBRARY  SPNAV_INCLUDE_DIR)

mark_as_advanced(SPNAV_INCLUDE_DIR SPNAV_LIBRARY)
