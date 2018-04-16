# - Try to find Open3D lib
#
# This module supports requiring a minimum version, e.g. you can do
#   find_package(Open3D 0.1.1)
# to require version 0.1.1 or newer of Open3D.
#
# Once done this will define
#
#  OPEN3D_FOUND - system has open3d lib with correct version
#  OPEN3D_INCLUDE_DIR - the open3d include directory
#  OPEN3D_LIBRARIES - the open3d library files
#  OPEN3D_VERSION - open3d version
#
# This module reads hints about search locations from
# the following enviroment variables:
#
# OPEN3D_ROOT
# OPEN3D_ROOT_DIR

# Copyright (c) 2018 Hamdi Sahloul <hamdisahloul@hotmail.com>
# Redistribution and use is allowed according to the terms of the 2-clause BSD license.

if(NOT Open3D_FIND_VERSION)
  if(NOT Open3D_FIND_VERSION_MAJOR)
    set(Open3D_FIND_VERSION_MAJOR 0)
  endif(NOT Open3D_FIND_VERSION_MAJOR)
  if(NOT Open3D_FIND_VERSION_MINOR)
    set(Open3D_FIND_VERSION_MINOR 1)
  endif(NOT Open3D_FIND_VERSION_MINOR)
  if(NOT Open3D_FIND_VERSION_PATCH)
    set(Open3D_FIND_VERSION_PATCH 0)
  endif(NOT Open3D_FIND_VERSION_PATCH)

  set(Open3D_FIND_VERSION "${Open3D_FIND_VERSION_MAJOR}.${Open3D_FIND_VERSION_MINOR}.${Open3D_FIND_VERSION_PATCH}")
endif(NOT Open3D_FIND_VERSION)

macro(_open3d_check_version)
  file(READ "${OPEN3D_INCLUDE_DIR}/Open3D/Core/Utility/Helper.h" _open3d_version_header)

  string(REGEX MATCH "define[ \t]+OPEN3D_VERSION_MAJOR[ \t]+([0-9]+)" _open3d_version_major_match "${_open3d_version_header}")
  set(OPEN3D_VERSION_MAJOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+OPEN3D_VERSION_MINOR[ \t]+([0-9]+)" _open3d_version_minor_match "${_open3d_version_header}")
  set(OPEN3D_VERSION_MINOR "${CMAKE_MATCH_1}")
  string(REGEX MATCH "define[ \t]+OPEN3D_VERSION_PATCH[ \t]+([0-9]+)" _open3d_version_patch_match "${_open3d_version_header}")
  set(OPEN3D_VERSION_PATCH "${CMAKE_MATCH_1}")

  set(OPEN3D_VERSION ${OPEN3D_VERSION_MAJOR}.${OPEN3D_VERSION_MINOR}.${OPEN3D_VERSION_PATCH})
  if(${OPEN3D_VERSION} VERSION_LESS ${Open3D_FIND_VERSION})
    set(OPEN3D_VERSION_OK FALSE)
  else(${OPEN3D_VERSION} VERSION_LESS ${Open3D_FIND_VERSION})
    set(OPEN3D_VERSION_OK TRUE)
  endif(${OPEN3D_VERSION} VERSION_LESS ${Open3D_FIND_VERSION})

  if(NOT OPEN3D_VERSION_OK)

    message(STATUS "Open3D version ${OPEN3D_VERSION} found in ${OPEN3D_INCLUDE_DIR}, "
                   "but at least version ${Open3D_FIND_VERSION} is required")
  endif(NOT OPEN3D_VERSION_OK)
endmacro(_open3d_check_version)

if (OPEN3D_INCLUDE_DIR)

  # in cache already
  _open3d_check_version()
  set(OPEN3D_FOUND ${OPEN3D_VERSION_OK})

else (OPEN3D_INCLUDE_DIR)

  # search first if an Open3DConfig.cmake is available in the system,
  # if successful this would set OPEN3D_INCLUDE_DIR and the rest of
  # the script will work as usual
  find_package(Open3D ${Open3D_FIND_VERSION} NO_MODULE QUIET)

  if(NOT OPEN3D_INCLUDE_DIR)
    find_path(OPEN3D_INCLUDE_DIR NAMES signature_of_open3d_library
        HINTS
        ENV OPEN3D_ROOT
        ENV OPEN3D_ROOT_DIR
        PATHS
        ${CMAKE_INSTALL_PREFIX}/include
        ${KDE4_INCLUDE_DIR}
        PATH_SUFFIXES open3d
      )
  endif(NOT OPEN3D_INCLUDE_DIR)

  if(OPEN3D_INCLUDE_DIR)
    _open3d_check_version()
  endif(OPEN3D_INCLUDE_DIR)

endif(OPEN3D_INCLUDE_DIR)

if(OPEN3D_INCLUDE_DIR AND NOT OPEN3D_LIBRARIES)

  find_library(OPEN3D_CORE_LIBRARY NAMES Core
      HINTS
      ENV OPEN3D_ROOT
      ENV OPEN3D_ROOT_DIR
      PATHS
      ${CMAKE_INSTALL_PREFIX}/lib
      ${OPEN3D_INCLUDE_DIR}/../lib
      PATH_SUFFIXES open3d
    )
  find_library(OPEN3D_IO_LIBRARY NAMES IO
      HINTS
      ENV OPEN3D_ROOT
      ENV OPEN3D_ROOT_DIR
      PATHS
      ${CMAKE_INSTALL_PREFIX}/lib
      ${OPEN3D_INCLUDE_DIR}/../lib
      PATH_SUFFIXES open3d
    )
  find_library(OPEN3D_VISUALIZATION_LIBRARY NAMES Visualization
      HINTS
      ENV OPEN3D_ROOT
      ENV OPEN3D_ROOT_DIR
      PATHS
      ${CMAKE_INSTALL_PREFIX}/lib
      ${OPEN3D_INCLUDE_DIR}/../lib
      PATH_SUFFIXES open3d
    )
  set(OPEN3D_LIBRARIES ${OPEN3D_CORE_LIBRARY} ${OPEN3D_IO_LIBRARY} ${OPEN3D_VISUALIZATION_LIBRARY} )

endif(OPEN3D_INCLUDE_DIR AND NOT OPEN3D_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Open3D DEFAULT_MSG OPEN3D_LIBRARIES OPEN3D_INCLUDE_DIR OPEN3D_VERSION_OK)
mark_as_advanced(OPEN3D_LIBRARIES OPEN3D_INCLUDE_DIR)
