# Try to find Mesa off-screen library and include dir.
# Once done this will define
#
# OSMESA_FOUND        - true if OSMesa has been found
# OSMESA_INCLUDE_DIR  - where the GL/osmesa.h can be found
# OSMESA_LIBRARY      - Link this to use OSMesa

if(NOT OSMESA_INCLUDE_DIR)

  # If we have a root defined look there first
  if(OSMESA_ROOT)
    find_path(OSMESA_INCLUDE_DIR GL/osmesa.h PATHS ${OSMESA_ROOT}/include
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT OSMESA_INCLUDE_DIR)
    find_path(OSMESA_INCLUDE_DIR GL/osmesa.h PATHS
      /usr/openwin/share/include
      /opt/graphics/OpenGL/include
    )
  endif()
endif()

# This may be left blank if OSMesa symbols are included
# in the main Mesa library
if(NOT OSMESA_LIBRARY)
  # If we have a root defined look there first
  if(OSMESA_ROOT)
    find_library(OSMESA_LIBRARY NAMES OSMesa OSMesa16 OSMesa32 PATHS ${OSMESA_ROOT}/lib
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT OSMESA_LIBRARY)
    find_library(OSMESA_LIBRARY NAMES OSMesa OSMesa16 OSMesa32 PATHS
      /opt/graphics/OpenGL/lib
      /usr/openwin/lib
    )
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OSMesa  DEFAULT_MSG  OSMESA_LIBRARY  OSMESA_INCLUDE_DIR)

mark_as_advanced(OSMESA_INCLUDE_DIR OSMESA_LIBRARY)
