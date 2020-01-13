###############################################################################
# Find FLANN
#
# This sets the following variables:
# FLANN_FOUND - True if FLANN was found.
# FLANN_INCLUDE_DIRS - Directories containing the FLANN include files.
# FLANN_LIBRARIES - Libraries needed to use FLANN.
# FLANN_DEFINITIONS - Compiler flags for FLANN.
# If FLANN_USE_STATIC is specified and then look for static libraries ONLY else
# look for shared ones

if(FLANN_USE_STATIC)
  set(FLANN_RELEASE_NAME flann_cpp_s)
  set(FLANN_DEBUG_NAME flann_cpp_s-gd)
else(FLANN_USE_STATIC)
  set(FLANN_RELEASE_NAME flann_cpp)
  set(FLANN_DEBUG_NAME flann_cpp-gd)
endif(FLANN_USE_STATIC)

find_package(PkgConfig QUIET)
if (FLANN_FIND_VERSION)
    pkg_check_modules(PC_FLANN flann>=${FLANN_FIND_VERSION})
else(FLANN_FIND_VERSION)
    pkg_check_modules(PC_FLANN flann)
endif(FLANN_FIND_VERSION)

set(FLANN_DEFINITIONS ${PC_FLANN_CFLAGS_OTHER})

find_path(FLANN_INCLUDE_DIR flann/flann.hpp
          HINTS ${PC_FLANN_INCLUDEDIR} ${PC_FLANN_INCLUDE_DIRS} "${FLANN_ROOT}" "$ENV{FLANN_ROOT}"
          PATHS "$ENV{PROGRAMFILES}/Flann" "$ENV{PROGRAMW6432}/Flann"
          PATH_SUFFIXES include)

find_library(FLANN_LIBRARY
             NAMES ${FLANN_RELEASE_NAME}
             HINTS ${PC_FLANN_LIBDIR} ${PC_FLANN_LIBRARY_DIRS} "${FLANN_ROOT}" "$ENV{FLANN_ROOT}"
             PATHS "$ENV{PROGRAMFILES}/Flann" "$ENV{PROGRAMW6432}/Flann"
	     PATH_SUFFIXES lib)

find_library(FLANN_LIBRARY_DEBUG
             NAMES ${FLANN_DEBUG_NAME} ${FLANN_RELEASE_NAME}
	     HINTS ${PC_FLANN_LIBDIR} ${PC_FLANN_LIBRARY_DIRS} "${FLANN_ROOT}" "$ENV{FLANN_ROOT}"
	     PATHS "$ENV{PROGRAMFILES}/Flann" "$ENV{PROGRAMW6432}/Flann"
	     PATH_SUFFIXES lib)

find_library(LZ4_LIBRARY
             NAMES lz4
       PATH_SUFFIXES lib)


if(NOT FLANN_LIBRARY_DEBUG)
  set(FLANN_LIBRARY_DEBUG ${FLANN_LIBRARY})
endif(NOT FLANN_LIBRARY_DEBUG)

set(FLANN_INCLUDE_DIRS ${FLANN_INCLUDE_DIR})
set(FLANN_LIBRARIES optimized ${FLANN_LIBRARY} debug ${FLANN_LIBRARY_DEBUG})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FLANN DEFAULT_MSG FLANN_LIBRARY FLANN_INCLUDE_DIR LZ4_LIBRARY)

mark_as_advanced(FLANN_LIBRARY FLANN_LIBRARY_DEBUG FLANN_INCLUDE_DIR)

if(FLANN_FOUND)
  message(STATUS "FLANN found (include: ${FLANN_INCLUDE_DIRS}, lib: ${FLANN_LIBRARIES}), lz4: ${LZ4_LIBRARY}")
  if(FLANN_USE_STATIC)
    add_definitions(-DFLANN_STATIC)
  endif(FLANN_USE_STATIC)
endif(FLANN_FOUND)
