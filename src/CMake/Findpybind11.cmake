# - Try to find Open3D lib
#
# This module supports requiring a minimum version, e.g. you can do
#   find_package(pybind11 0.1.1)
# to require version 0.1.1 or newer of pybind11.
#
# Once done this will define
#
#  PYBIND11_FOUND - system has pybind11 lib with correct version
#  PYBIND11_INCLUDE_DIR - the pybind11 include directory
##  PYBIND11_CMAKE_TOOLS - pybind11 CMake tools file
#  pybind11_VERSION - pybind11 version
#
# This module reads hints about search locations from
# the following enviroment variables:
#
# PYBIND11_ROOT
# PYBIND11_ROOT_DIR

# Copyright (c) 2018 Hamdi Sahloul <hamdisahloul@hotmail.com>
# Redistribution and use is allowed according to the terms of the 2-clause BSD license.

if(NOT PYBIND11_FIND_VERSION)
  if(NOT PYBIND11_FIND_VERSION_MAJOR)
    set(PYBIND11_FIND_VERSION_MAJOR 0)
  endif(NOT PYBIND11_FIND_VERSION_MAJOR)
  if(NOT PYBIND11_FIND_VERSION_MINOR)
    set(PYBIND11_FIND_VERSION_MINOR 1)
  endif(NOT PYBIND11_FIND_VERSION_MINOR)
  if(NOT PYBIND11_FIND_VERSION_PATCH)
    set(PYBIND11_FIND_VERSION_PATCH 0)
  endif(NOT PYBIND11_FIND_VERSION_PATCH)

  set(PYBIND11_FIND_VERSION "${PYBIND11_FIND_VERSION_MAJOR}.${PYBIND11_FIND_VERSION_MINOR}.${PYBIND11_FIND_VERSION_PATCH}")
endif(NOT PYBIND11_FIND_VERSION)

macro(_pybind11_check_version)
  file(STRINGS "${PYBIND11_INCLUDE_DIR}/pybind11/detail/common.h" pybind11_version_defines
       REGEX "#define PYBIND11_VERSION_(MAJOR|MINOR|PATCH) ")
  foreach(ver ${pybind11_version_defines})
    if (ver MATCHES "#define PYBIND11_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
      set(PYBIND11_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
  endforeach()
  set(pybind11_VERSION ${PYBIND11_VERSION_MAJOR}.${PYBIND11_VERSION_MINOR}.${PYBIND11_VERSION_PATCH})  
  
  if(${pybind11_VERSION} VERSION_LESS ${PYBIND11_FIND_VERSION})
    set(PYBIND11_VERSION_OK FALSE)
  else(${pybind11_VERSION} VERSION_LESS ${PYBIND11_FIND_VERSION})
    set(PYBIND11_VERSION_OK TRUE)
  endif(${pybind11_VERSION} VERSION_LESS ${PYBIND11_FIND_VERSION})

  if(NOT PYBIND11_VERSION_OK)

    message(STATUS "pybind11 version ${pybind11_VERSION} found in ${PYBIND11_INCLUDE_DIR}, "
                   "but at least version ${PYBIND11_FIND_VERSION} is required")
  endif(NOT PYBIND11_VERSION_OK)
endmacro(_pybind11_check_version)

if (PYBIND11_INCLUDE_DIR)

  # in cache already
  _pybind11_check_version()
  set(PYBIND11_FOUND ${PYBIND11_VERSION_OK})

else (PYBIND11_INCLUDE_DIR)

  # search first if an pybind11Config.cmake is available in the system,
  # if successful this would set PYBIND11_INCLUDE_DIR and the rest of
  # the script will work as usual
  find_package(pybind11 ${PYBIND11_FIND_VERSION} NO_MODULE QUIET)

  if(NOT PYBIND11_INCLUDE_DIR)
    find_path(PYBIND11_INCLUDE_DIR NAMES signature_of_pybind11_library
        HINTS
        ENV PYBIND11_ROOT
        ENV PYBIND11_ROOT_DIR
        PATHS
        ${CMAKE_INSTALL_PREFIX}/include
        ${KDE4_INCLUDE_DIR}
        PATH_SUFFIXES pybind11
      )
  endif(NOT PYBIND11_INCLUDE_DIR)

  if(PYBIND11_INCLUDE_DIR)
    _pybind11_check_version()
  endif(PYBIND11_INCLUDE_DIR)

endif(PYBIND11_INCLUDE_DIR)

SET(PYBIND11_INCLUDE_DIR ${PYBIND11_INCLUDE_DIR} CACHE PATH "The path to pybind11 headers" FORCE)
#get_filename_component(PYBIND11_CMAKE_TOOLS "${PYBIND11_INCLUDE_DIR}/../share/cmake/pybind11/pybind11Tools.cmake" ABSOLUTE)
#SET(PYBIND11_CMAKE_TOOLS ${PYBIND11_CMAKE_TOOLS} CACHE FILEPATH "The path to ${LIB_NAME} CMake tools file" FORCE)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(pybind11
  FOUND_VAR PYBIND11_FOUND
  REQUIRED_VARS PYBIND11_INCLUDE_DIR) #PYBIND11_CMAKE_TOOLS
mark_as_advanced(PYBIND11_INCLUDE_DIR) #PYBIND11_CMAKE_TOOLS
