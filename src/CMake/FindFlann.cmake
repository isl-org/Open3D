#.rst:
# FindFLANN
# ---------
#
# Find Fast Library for Approximate Nearest Neighbors (FLANN).
# See http://www.cs.ubc.ca/research/flann for details.
#
# This module considers the following CMake variables set by find_package:
#
# ::
#
#    FLANN_FIND_COMPONENTS   - Names of requested components:
#                                cpp        - shared or static C++ library
#                                cpp_shared - shared C++ library
#                                cpp_static - static C++ library
#                                c          - shared or static C bindings library
#                                c_shared   - shared C bindings library
#                                c_static   - static C bindings library
#                                matlab     - MATLAB bindings
#                                python     - Python bindings
#    FLANN_FIND_REQUIRED_<C> - Whether FLANN component <C> is required.
#                              FLANN is considered to be not found when at least
#                              one required library or its include path is missing.
#                              When no FLANN_FIND_COMPONENTS are specified,
#                              the static and shared C++ libraries are looked for.
#    FLANN_FIND_REQUIRED     - Raise FATAL_ERROR when required components not found.
#    FLANN_FIND_QUIETLY      - Suppress all other (status) messages.
#
# .. note::
#
#    The "matlab" and "python" components are currently not supported yet.
#
# This module caches the following variables:
#
# ::
#
#    FLANN_INCLUDE_DIR        - Include path of C/C++ header files.
#    FLANN_C_LIBRARY_SHARED   - Path of shared C bindings link library.
#    FLANN_C_LIBRARY_STATIC   - Path of static C bindings link library.
#    FLANN_CPP_LIBRARY_SHARED - Path of shared C++ link library.
#    FLANN_CPP_LIBRARY_STATIC - Path of static c++ link library.
#
# It further defines the following uncached variables:
#
# ::
#
#    FLANN_FOUND          - Whether all required FLANN components were found.
#    FLANN_<C>_FOUND      - Whether library component <C> was found.
#    FLANN_C_LIBRARY      - Path of C bindings link library (shared preferred).
#    FLANN_CPP_LIBRARY    - Path of C++ link library (shared preferred).
#    FLANN_LIBRARIES      - Paths of all found libraries (shared preferred).
#    FLANN_VERSION        - Version for use in VERSION_LESS et al. comparisons.
#    FLANN_VERSION_MAJOR  - Major library version number.
#    FLANN_VERSION_MINOR  - Minor library version number.
#    FLANN_VERSION_PATCH  - Patch library version number.
#    FLANN_VERSION_STRING - Version string for output messages.

#=============================================================================
# Copyright 2016 Andreas Schuh <andreas.schuh.84@gmail.com>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

if (NOT FLANN_FIND_QUIETLY)
set(_FLANN_FIND_STATUS "Looking for FLANN")
if (FLANN_FIND_COMPONENTS)
set(_FLANN_FIND_STATUS "${_FLANN_FIND_STATUS} [${FLANN_FIND_COMPONENTS}]")
endif ()
if (NOT FLANN_FIND_REQUIRED)
set(_FLANN_FIND_STATUS "${_FLANN_FIND_STATUS} (optional)")
endif ()
message(STATUS "${_FLANN_FIND_STATUS}...")
endif ()

# ------------------------------------------------------------------------------
# Components
set(_FLANN_COMPONENTS c c_shared c_static cpp cpp_shared cpp_static matlab python)
foreach (_FLANN_COMPONENT IN LISTS _FLANN_COMPONENTS)
set(_FLANN_FIND_${_FLANN_COMPONENT} FALSE)
endforeach ()

if (NOT FLANN_FIND_COMPONENTS)
set(FLANN_FIND_COMPONENTS cpp)
set(FLANN_FIND_REQUIRED_cpp ${FLANN_FIND_REQUIRED})
endif ()

foreach (_FLANN_COMPONENT IN LISTS FLANN_FIND_COMPONENTS)
if     (_FLANN_COMPONENT MATCHES "^c$")
set(_FLANN_FIND_c TRUE)
elseif (_FLANN_COMPONENT MATCHES "^(c_shared|flann)$")
set(_FLANN_FIND_c_shared TRUE)
elseif (_FLANN_COMPONENT MATCHES "^(c_static|flann_s)$")
set(_FLANN_FIND_c_static TRUE)
elseif (_FLANN_COMPONENT MATCHES "^cpp$")
set(_FLANN_FIND_cpp TRUE)
elseif (_FLANN_COMPONENT MATCHES "^(cpp_shared|flann_cpp)$")
set(_FLANN_FIND_cpp_shared TRUE)
elseif (_FLANN_COMPONENT MATCHES "^(cpp_static|cpp_s|flann_cpp_s)$")
set(_FLANN_FIND_cpp_static TRUE)
elseif (_FLANN_COMPONENT MATCHES "^(matlab|mex)$")
message(FATAL_ERROR "FLANN library component \"${_FLANN_COMPONENT}\" not supported yet")
elseif (_FLANN_COMPONENT MATCHES "^python$")
message(FATAL_ERROR "FLANN library component \"python\" not supported yet")
else ()
message(FATAL_ERROR "Unknown FLANN library component: ${_FLANN_COMPONENT}\n"
"Valid component names are: c, c_shared|flann, c_static|flann_s, cpp_shared|flann_cpp, cpp_static|cpp_s|flann_cpp_s, matlab|mex, python")
endif ()
endforeach ()

if (FLANN_DEBUG)
message("** FindFLANN: Components = [${FLANN_FIND_COMPONENTS}]")
foreach (_FLANN_COMPONENT IN LISTS _FLANN_COMPONENTS)
message("** FindFLANN: - Find component ${_FLANN_COMPONENT} = ${_FLANN_FIND_${_FLANN_COMPONENT}}")
endforeach ()
endif ()

# ------------------------------------------------------------------------------
# Construct a set of search paths
set(_FLANN_INC_DIR_HINTS)
set(_FLANN_LIB_DIR_HINTS)

if (NOT FLANN_ROOT)
file(TO_CMAKE_PATH "$ENV{FLANN_ROOT}" FLANN_ROOT)
endif ()

find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
pkg_check_modules(_FLANN flann QUIET)
if (_FLANN_INCLUDEDIR)
list(APPEND _FLANN_INC_DIR_HINTS ${_FLANN_INCLUDEDIR})
endif ()
if (_FLANN_INCLUDE_DIRS)
list(APPEND _FLANN_INC_DIR_HINTS ${_FLANN_INCLUDE_DIRS})
endif ()
if (_FLANN_LIBDIR)
list(APPEND _FLANN_LIB_DIR_HINTS ${_FLANN_LIBDIR})
endif ()
if (_FLANN_CPP_LIBRARY_STATIC_DIRS)
list(APPEND _FLANN_LIB_DIR_HINTS ${_FLANN_CPP_LIBRARY_STATIC_DIRS})
endif ()
unset(_FLANN_INCLUDEDIR)
unset(_FLANN_INCLUDE_DIRS)
unset(_FLANN_LIBDIR)
unset(_FLANN_CPP_LIBRARY_STATIC_DIRS)
unset(_FLANN_CFLAGS_OTHER)
endif ()

if (FLANN_DEBUG)
message("** FindFLANN: Initial search paths:")
message("** FindFLANN: - Root directory hints   = [${FLANN_ROOT}]")
message("** FindFLANN: - PkgConfig include path = [${_FLANN_INC_DIR_HINTS}]")
message("** FindFLANN: - PkgConfig library path = [${_FLANN_LIB_DIR_HINTS}]")
endif ()

# ------------------------------------------------------------------------------
# Find common include directory
#
# Looking for flann/config.h because we use this path later to read this file
# in order to extract the version information.
find_path(FLANN_INCLUDE_DIR
NAMES flann/config.h
HINTS ${FLANN_ROOT} ${_FLANN_INC_DIR_HINTS}
)

mark_as_advanced(FLANN_INCLUDE_DIR)

# ------------------------------------------------------------------------------
# Derive FLANN_ROOT from FLANN_INCLUDE_DIR if unset
if (FLANN_INCLUDE_DIR AND NOT FLANN_ROOT)
get_filename_component(FLANN_ROOT "${FLANN_INCLUDE_DIR}" DIRECTORY)
endif ()

if (FLANN_DEBUG)
message("** FindFLANN: After initial search of FLANN include path")
message("** FindFLANN: - FLANN_INCLUDE_DIR = ${FLANN_INCLUDE_DIR}")
message("** FindFLANN: - FLANN_ROOT        = [${FLANN_ROOT}]")
endif ()

# ------------------------------------------------------------------------------
# Find libraries
unset(FLANN_C_LIBRARY)
unset(FLANN_CPP_LIBRARY)

set(FLANN_INCLUDE_DIRS ${FLANN_INCLUDE_DIR})
set(FLANN_LIBRARIES)

foreach (_FLANN_COMPONENT IN LISTS FLANN_FIND_COMPONENTS)
set(FLANN_${_FLANN_COMPONENT}_FOUND FALSE)
endforeach ()

if (FLANN_INCLUDE_DIR)

# Shared C library
if (_FLANN_FIND_c OR _FLANN_FIND_c_shared)
find_library(FLANN_C_LIBRARY_SHARED
NAMES flann
HINTS ${FLANN_ROOT} ${_FLANN_LIB_DIR_HINTS}
)
if (FLANN_C_LIBRARY_SHARED)
set(FLANN_c_FOUND        TRUE)
set(FLANN_c_shared_FOUND TRUE)
endif ()
mark_as_advanced(FLANN_C_LIBRARY_SHARED)
endif ()

# Static C library
if (_FLANN_FIND_c OR _FLANN_FIND_c_static)
find_library(FLANN_C_LIBRARY_STATIC
NAMES flann_s
HINTS ${FLANN_ROOT} ${_FLANN_LIB_DIR_HINTS}
)
if (FLANN_C_LIBRARY_STATIC)
set(FLANN_c_FOUND        TRUE)
set(FLANN_c_static_FOUND TRUE)
endif ()
mark_as_advanced(FLANN_C_LIBRARY_STATIC)
endif ()

# Set FLANN_C_LIBRARY and add it to FLANN_LIBRARIES
if (FLANN_C_LIBRARY_SHARED)
set(FLANN_C_LIBRARY ${FLANN_C_LIBRARY_SHARED})
elseif (FLANN_C_LIBRARY_STATIC)
set(FLANN_C_LIBRARY ${FLANN_C_LIBRARY_STATIC})
endif ()
if (FLANN_C_LIBRARY)
list(APPEND FLANN_LIBRARIES ${FLANN_C_LIBRARY})
endif ()

# Shared C++ library
if (_FLANN_FIND_cpp OR _FLANN_FIND_cpp_shared)
find_library(FLANN_CPP_LIBRARY_SHARED
NAMES flann_cpp
HINTS ${FLANN_ROOT} ${_FLANN_LIB_DIR_HINTS}
)
if (FLANN_CPP_LIBRARY_SHARED)
set(FLANN_cpp_FOUND        TRUE)
set(FLANN_cpp_shared_FOUND TRUE)
endif ()
mark_as_advanced(FLANN_CPP_LIBRARY_SHARED)
endif ()

# Static C++ library
if (_FLANN_FIND_cpp OR _FLANN_FIND_cpp_static)
find_library(FLANN_CPP_LIBRARY_STATIC
NAMES flann_cpp_s
HINTS ${FLANN_ROOT} ${_FLANN_LIB_DIR_HINTS}
)
if (FLANN_CPP_LIBRARY_STATIC)
set(FLANN_cpp_FOUND        TRUE)
set(FLANN_cpp_static_FOUND TRUE)
endif ()
mark_as_advanced(FLANN_CPP_LIBRARY_STATIC)
endif ()

# Set FLANN_CPP_LIBRARY and add it to FLANN_LIBRARIES
if (FLANN_CPP_LIBRARY_SHARED)
set(FLANN_CPP_LIBRARY ${FLANN_CPP_LIBRARY_SHARED})
elseif (FLANN_CPP_LIBRARY_STATIC)
set(FLANN_CPP_LIBRARY ${FLANN_CPP_LIBRARY_STATIC})
endif ()
if (FLANN_CPP_LIBRARY)
list(APPEND FLANN_LIBRARIES ${FLANN_CPP_LIBRARY})
endif ()

if (FLANN_DEBUG)
message("** FindFLANN: C/C++ library paths:")
message("** FindFLANN: - FLANN_C_LIBRARY          = ${FLANN_C_LIBRARY}")
message("** FindFLANN: - FLANN_C_LIBRARY_SHARED   = ${FLANN_C_LIBRARY_SHARED}")
message("** FindFLANN: - FLANN_C_LIBRARY_STATIC   = ${FLANN_C_LIBRARY_STATIC}")
message("** FindFLANN: - FLANN_CPP_LIBRARY        = ${FLANN_CPP_LIBRARY}")
message("** FindFLANN: - FLANN_CPP_LIBRARY_SHARED = ${FLANN_CPP_LIBRARY_SHARED}")
message("** FindFLANN: - FLANN_CPP_LIBRARY_STATIC = ${FLANN_CPP_LIBRARY_STATIC}")
message("** FindFLANN: - FLANN_LIBRARIES          = [${FLANN_LIBRARIES}]")
endif ()
endif ()

# ------------------------------------------------------------------------------
# Extract library version from flann/config.h
if (FLANN_INCLUDE_DIR)
if (NOT DEFINED FLANN_VERSION_MAJOR OR
NOT DEFINED FLANN_VERSION_MINOR OR
NOT DEFINED FLANN_VERSION_PATCH)
file(READ "${FLANN_INCLUDE_DIR}/flann/config.h" _FLANN_CONFIG_CONTENTS LIMIT 2048)
if (_FLANN_CONFIG_CONTENTS MATCHES "#define FLANN_VERSION_? \"([0-9]+)\\.([0-9]+)\\.([0-9]+)\"")
set(FLANN_VERSION_MAJOR ${CMAKE_MATCH_1})
set(FLANN_VERSION_MINOR ${CMAKE_MATCH_2})
set(FLANN_VERSION_PATCH ${CMAKE_MATCH_3})
else ()
if (NOT FLANN_FIND_QUIETLY)
message(WARNING "Could not extract FLANN version numbers from: ${FLANN_INCLUDE_DIR}/flann/config.h")
endif ()
set(FLANN_VERSION_MAJOR 0)
set(FLANN_VERSION_MINOR 0)
set(FLANN_VERSION_PATCH 0)
endif ()
unset(_FLANN_CONFIG_CONTENTS)
endif ()
set(FLANN_VERSION "${FLANN_VERSION_MAJOR}.${FLANN_VERSION_MINOR}.${FLANN_VERSION_PATCH}")
set(FLANN_VERSION_STRING "${FLANN_VERSION}")
else ()
unset(FLANN_VERSION)
unset(FLANN_VERSION_MAJOR)
unset(FLANN_VERSION_MINOR)
unset(FLANN_VERSION_PATCH)
unset(FLANN_VERSION_STRING)
endif ()

if (FLANN_DEBUG)
message("** FindFLANN: Version information from ${FLANN_INCLUDE_DIR}/flann/config.h")
message("** FindFLANN: - FLANN_VERSION_STRING = ${FLANN_VERSION_STRING}")
message("** FindFLANN: - FLANN_VERSION_MAJOR  = ${FLANN_VERSION_MAJOR}")
message("** FindFLANN: - FLANN_VERSION_MINOR  = ${FLANN_VERSION_MINOR}")
message("** FindFLANN: - FLANN_VERSION_PATCH  = ${FLANN_VERSION_PATCH}")
endif ()

# ------------------------------------------------------------------------------
# Handle QUIET, REQUIRED, and [EXACT] VERSION arguments and set FLANN_FOUND
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FLANN
REQUIRED_VARS FLANN_INCLUDE_DIR
VERSION_VAR   FLANN_VERSION
HANDLE_COMPONENTS
)

if (NOT FLANN_FIND_QUIETLY)
if (FLANN_FOUND)
message(STATUS "${_FLANN_FIND_STATUS}... - found v${FLANN_VERSION_STRING}")
else ()
message(STATUS "${_FLANN_FIND_STATUS}... - not found")
endif ()
endif ()

# ------------------------------------------------------------------------------
# Unset local auxiliary variables
foreach (_FLANN_COMPONENT IN LISTS _FLANN_COMPONENTS)
unset(_FLANN_FIND_${_FLANN_COMPONENT})
endforeach ()
unset(_FLANN_COMPONENT)
unset(_FLANN_COMPONENTS)
unset(_FLANN_FIND_STATUS)
