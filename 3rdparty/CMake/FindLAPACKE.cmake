###
#
# @copyright (c) 2009-2014 The University of Tennessee and The University
#                          of Tennessee Research Foundation.
#                          All rights reserved.
# @copyright (c) 2012-2016 Inria. All rights reserved.
# @copyright (c) 2012-2014 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria, Univ. Bordeaux. All rights reserved.
#
###
#
# - Find LAPACKE include dirs and libraries
# Use this module by invoking find_package with the form:
#  find_package(LAPACKE
#               [REQUIRED] # Fail with error if lapacke is not found
#               [COMPONENTS <comp1> <comp2> ...] # dependencies
#              )
#
#  LAPACKE depends on the following libraries:
#   - LAPACK
#
# This module finds headers and lapacke library.
# Results are reported in variables:
#  LAPACKE_FOUND            - True if headers and requested libraries were found
#  LAPACKE_LINKER_FLAGS     - list of required linker flags (excluding -l and -L)
#  LAPACKE_INCLUDE_DIRS     - lapacke include directories
#  LAPACKE_LIBRARY_DIRS     - Link directories for lapacke libraries
#  LAPACKE_LIBRARIES        - lapacke component libraries to be linked
#  LAPACKE_INCLUDE_DIRS_DEP - lapacke + dependencies include directories
#  LAPACKE_LIBRARY_DIRS_DEP - lapacke + dependencies link directories
#  LAPACKE_LIBRARIES_DEP    - lapacke libraries + dependencies
#
# The user can give specific paths where to find the libraries adding cmake
# options at configure (ex: cmake path/to/project -DLAPACKE_DIR=path/to/lapacke):
#  LAPACKE_DIR             - Where to find the base directory of lapacke
#  LAPACKE_INCDIR          - Where to find the header files
#  LAPACKE_LIBDIR          - Where to find the library files
# The module can also look for the following environment variables if paths
# are not given as cmake variable: LAPACKE_DIR, LAPACKE_INCDIR, LAPACKE_LIBDIR
#
# If the static version of the LAPACKE libraries is required, please add the
# following in your CMakeLists.txt before calling find_package(LAPACKE):
# set(LAPACKE_STATIC TRUE)
#
# LAPACKE could be directly embedded in LAPACK library (ex: Intel MKL) so that
# we test a lapacke function with the lapack libraries found and set LAPACKE
# variables to LAPACK ones if test is successful. To skip this feature and
# look for a stand alone lapacke, please add the following in your
# CMakeLists.txt before to call find_package(LAPACKE):
# set(LAPACKE_STANDALONE TRUE)

#=============================================================================
# Copyright 2012-2013 Inria
# Copyright 2012-2013 Emmanuel Agullo
# Copyright 2012-2013 Mathieu Faverge
# Copyright 2012      Cedric Castagnede
# Copyright 2013-2016 Florent Pruvost
#
# This file is part of a computer program whose purpose is to process
# Matrices Over Runtime Systems @ Exascale (MORSE). More information
# can be found on the following website: http://www.inria.fr/en/teams/morse.
#
# This software is governed by the CeCILL-C license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL-C
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-C license and that you accept its terms.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================

if (NOT LAPACKE_FOUND)
  set(LAPACKE_DIR "" CACHE PATH "Installation directory of LAPACKE library")
  if (NOT LAPACKE_FIND_QUIETLY)
    message(STATUS "A cache variable, namely LAPACKE_DIR, has been set to specify the install directory of LAPACKE")
  endif()
endif()

# LAPACKE depends on LAPACK anyway, try to find it
if (NOT LAPACK_FOUND)
  if(LAPACKE_FIND_REQUIRED)
    find_package(LAPACKEXT REQUIRED)
  else()
    find_package(LAPACKEXT)
  endif()
endif()

# LAPACKE depends on LAPACK
if (LAPACK_FOUND)

  if (NOT LAPACKE_STANDALONE)
    # check if a lapacke function exists in the LAPACK lib
    include(CheckFunctionExists)
    set(CMAKE_REQUIRED_LIBRARIES "${LAPACK_LINKER_FLAGS};${LAPACK_LIBRARIES}")
    unset(LAPACKE_WORKS CACHE)
    check_function_exists(LAPACKE_dgeqrf LAPACKE_WORKS)
    mark_as_advanced(LAPACKE_WORKS)
    set(CMAKE_REQUIRED_LIBRARIES)

    if(LAPACKE_WORKS)
      if(NOT LAPACKE_FIND_QUIETLY)
        message(STATUS "Looking for lapacke: test with lapack succeeds")
      endif()
      # test succeeds: LAPACKE is in LAPACK
      set(LAPACKE_LIBRARIES "${LAPACK_LIBRARIES}")
      set(LAPACKE_LIBRARIES_DEP "${LAPACK_LIBRARIES}")
      if (LAPACK_LIBRARY_DIRS)
        set(LAPACKE_LIBRARY_DIRS "${LAPACK_LIBRARY_DIRS}")
      endif()
      if(LAPACK_INCLUDE_DIRS)
        set(LAPACKE_INCLUDE_DIRS "${LAPACK_INCLUDE_DIRS}")
        set(LAPACKE_INCLUDE_DIRS_DEP "${LAPACK_INCLUDE_DIRS}")
      endif()
      if (LAPACK_LINKER_FLAGS)
        set(LAPACKE_LINKER_FLAGS "${LAPACK_LINKER_FLAGS}")
      endif()
    endif()
  endif (NOT LAPACKE_STANDALONE)

  if (LAPACKE_STANDALONE OR NOT LAPACKE_WORKS)

    if(NOT LAPACKE_WORKS AND NOT LAPACKE_FIND_QUIETLY)
      message(STATUS "Looking for lapacke : test with lapack fails")
    endif()
    # test fails: try to find LAPACKE lib exterior to LAPACK

    # Try to find LAPACKE lib
    #######################

    # Looking for include
    # -------------------

    # Add system include paths to search include
    # ------------------------------------------
    unset(_inc_env)
    set(ENV_LAPACKE_DIR "$ENV{LAPACKE_DIR}")
    set(ENV_LAPACKE_INCDIR "$ENV{LAPACKE_INCDIR}")
    if(ENV_LAPACKE_INCDIR)
      list(APPEND _inc_env "${ENV_LAPACKE_INCDIR}")
    elseif(ENV_LAPACKE_DIR)
      list(APPEND _inc_env "${ENV_LAPACKE_DIR}")
      list(APPEND _inc_env "${ENV_LAPACKE_DIR}/include")
      list(APPEND _inc_env "${ENV_LAPACKE_DIR}/include/lapacke")
    else()
      if(WIN32)
        string(REPLACE ":" ";" _inc_env "$ENV{INCLUDE}")
      else()
        string(REPLACE ":" ";" _path_env "$ENV{INCLUDE}")
        list(APPEND _inc_env "${_path_env}")
        string(REPLACE ":" ";" _path_env "$ENV{C_INCLUDE_PATH}")
        list(APPEND _inc_env "${_path_env}")
        string(REPLACE ":" ";" _path_env "$ENV{CPATH}")
        list(APPEND _inc_env "${_path_env}")
        string(REPLACE ":" ";" _path_env "$ENV{INCLUDE_PATH}")
        list(APPEND _inc_env "${_path_env}")
      endif()
    endif()
    list(APPEND _inc_env "${CMAKE_PLATFORM_IMPLICIT_INCLUDE_DIRECTORIES}")
    list(APPEND _inc_env "${CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES}")
    list(REMOVE_DUPLICATES _inc_env)


    # Try to find the lapacke header in the given paths
    # -------------------------------------------------
    # call cmake macro to find the header path
    if(LAPACKE_INCDIR)
      set(LAPACKE_lapacke.h_DIRS "LAPACKE_lapacke.h_DIRS-NOTFOUND")
      find_path(LAPACKE_lapacke.h_DIRS
        NAMES lapacke.h
        HINTS ${LAPACKE_INCDIR})
    else()
      if(LAPACKE_DIR)
        set(LAPACKE_lapacke.h_DIRS "LAPACKE_lapacke.h_DIRS-NOTFOUND")
        find_path(LAPACKE_lapacke.h_DIRS
          NAMES lapacke.h
          HINTS ${LAPACKE_DIR}
          PATH_SUFFIXES "include" "include/lapacke")
      else()
        set(LAPACKE_lapacke.h_DIRS "LAPACKE_lapacke.h_DIRS-NOTFOUND")
        find_path(LAPACKE_lapacke.h_DIRS
          NAMES lapacke.h
          HINTS ${_inc_env})
      endif()
    endif()
    mark_as_advanced(LAPACKE_lapacke.h_DIRS)

    # If found, add path to cmake variable
    # ------------------------------------
    if (LAPACKE_lapacke.h_DIRS)
      set(LAPACKE_INCLUDE_DIRS "${LAPACKE_lapacke.h_DIRS}")
    else ()
      set(LAPACKE_INCLUDE_DIRS "LAPACKE_INCLUDE_DIRS-NOTFOUND")
      if(NOT LAPACKE_FIND_QUIETLY)
        message(STATUS "Looking for lapacke -- lapacke.h not found")
      endif()
    endif()


    # Looking for lib
    # ---------------

    # Add system library paths to search lib
    # --------------------------------------
    unset(_lib_env)
    set(ENV_LAPACKE_LIBDIR "$ENV{LAPACKE_LIBDIR}")
    if(ENV_LAPACKE_LIBDIR)
      list(APPEND _lib_env "${ENV_LAPACKE_LIBDIR}")
    elseif(ENV_LAPACKE_DIR)
      list(APPEND _lib_env "${ENV_LAPACKE_DIR}")
      list(APPEND _lib_env "${ENV_LAPACKE_DIR}/lib")
    else()
      if(WIN32)
        string(REPLACE ":" ";" _lib_env "$ENV{LIB}")
      else()
        if(APPLE)
          string(REPLACE ":" ";" _lib_env "$ENV{DYLD_LIBRARY_PATH}")
        else()
          string(REPLACE ":" ";" _lib_env "$ENV{LD_LIBRARY_PATH}")
        endif()
        list(APPEND _lib_env "${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES}")
        list(APPEND _lib_env "${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}")
      endif()
    endif()
    list(REMOVE_DUPLICATES _lib_env)

    # Try to find the lapacke lib in the given paths
    # ----------------------------------------------

    # name of the lapacke library
    set(LAPACKE_lapacke_NAMES "lapacke")
    if(LAPACKE_STATIC)
        if(WIN32)
            set(LAPACKE_lapacke_NAMES "liblapacke.lib")
        endif()

        if(UNIX)
            set(LAPACKE_lapacke_NAMES "liblapacke.a")
        endif()
    endif()

    # call cmake macro to find the lib path
    if(LAPACKE_LIBDIR)
      set(LAPACKE_lapacke_LIBRARY "LAPACKE_lapacke_LIBRARY-NOTFOUND")
      find_library(LAPACKE_lapacke_LIBRARY
        NAMES ${LAPACKE_lapacke_NAMES}
        HINTS ${LAPACKE_LIBDIR})
    else()
      if(LAPACKE_DIR)
        set(LAPACKE_lapacke_LIBRARY "LAPACKE_lapacke_LIBRARY-NOTFOUND")
        find_library(LAPACKE_lapacke_LIBRARY
          NAMES ${LAPACKE_lapacke_NAMES}
          HINTS ${LAPACKE_DIR}
          PATH_SUFFIXES lib lib32 lib64)
      else()
        set(LAPACKE_lapacke_LIBRARY "LAPACKE_lapacke_LIBRARY-NOTFOUND")
        find_library(LAPACKE_lapacke_LIBRARY
          NAMES ${LAPACKE_lapacke_NAMES}
          HINTS ${_lib_env})
      endif()
    endif()
    mark_as_advanced(LAPACKE_lapacke_LIBRARY)

    # If found, add path to cmake variable
    # ------------------------------------
    if (LAPACKE_lapacke_LIBRARY)
      get_filename_component(lapacke_lib_path "${LAPACKE_lapacke_LIBRARY}" PATH)
      # set cmake variables
      set(LAPACKE_LIBRARIES    "${LAPACKE_lapacke_LIBRARY}")
      set(LAPACKE_LIBRARY_DIRS "${lapacke_lib_path}")
    else ()
      set(LAPACKE_LIBRARIES    "LAPACKE_LIBRARIES-NOTFOUND")
      set(LAPACKE_LIBRARY_DIRS "LAPACKE_LIBRARY_DIRS-NOTFOUND")
      if (NOT LAPACKE_FIND_QUIETLY)
        message(STATUS "Looking for lapacke -- lib lapacke not found")
      endif()
    endif ()

    # check a function to validate the find
    if(LAPACKE_LIBRARIES)

      set(REQUIRED_LDFLAGS)
      set(REQUIRED_INCDIRS)
      set(REQUIRED_LIBDIRS)
      set(REQUIRED_LIBS)

      # LAPACKE
      if (LAPACKE_INCLUDE_DIRS)
        set(REQUIRED_INCDIRS "${LAPACKE_INCLUDE_DIRS}")
      endif()
      if (LAPACKE_LIBRARY_DIRS)
        set(REQUIRED_LIBDIRS "${LAPACKE_LIBRARY_DIRS}")
      endif()
      set(REQUIRED_LIBS "${LAPACKE_LIBRARIES}")
      # LAPACK
      if (LAPACK_INCLUDE_DIRS)
        list(APPEND REQUIRED_INCDIRS "${LAPACK_INCLUDE_DIRS}")
      endif()
      if (LAPACK_LIBRARY_DIRS)
        list(APPEND REQUIRED_LIBDIRS "${LAPACK_LIBRARY_DIRS}")
      endif()
      list(APPEND REQUIRED_LIBS "${LAPACK_LIBRARIES}")
      if (LAPACK_LINKER_FLAGS)
        list(APPEND REQUIRED_LDFLAGS "${LAPACK_LINKER_FLAGS}")
      endif()
      # Fortran
      if (CMAKE_C_COMPILER_ID MATCHES "GNU")
        find_library(
          FORTRAN_gfortran_LIBRARY
          NAMES gfortran
          HINTS ${_lib_env}
          )
        mark_as_advanced(FORTRAN_gfortran_LIBRARY)
        if (FORTRAN_gfortran_LIBRARY)
          list(APPEND REQUIRED_LIBS "${FORTRAN_gfortran_LIBRARY}")
        endif()
      elseif (CMAKE_C_COMPILER_ID MATCHES "Intel")
        find_library(
          FORTRAN_ifcore_LIBRARY
          NAMES ifcore
          HINTS ${_lib_env}
          )
        mark_as_advanced(FORTRAN_ifcore_LIBRARY)
        if (FORTRAN_ifcore_LIBRARY)
          list(APPEND REQUIRED_LIBS "${FORTRAN_ifcore_LIBRARY}")
        endif()
      endif()
      # m
      find_library(M_LIBRARY NAMES m HINTS ${_lib_env})
      mark_as_advanced(M_LIBRARY)
      if(M_LIBRARY)
        list(APPEND REQUIRED_LIBS "-lm")
      endif()
      # set required libraries for link
      set(CMAKE_REQUIRED_INCLUDES "${REQUIRED_INCDIRS}")
      set(CMAKE_REQUIRED_LIBRARIES)
      list(APPEND CMAKE_REQUIRED_LIBRARIES "${REQUIRED_LDFLAGS}")
      foreach(lib_dir ${REQUIRED_LIBDIRS})
        list(APPEND CMAKE_REQUIRED_LIBRARIES "-L${lib_dir}")
      endforeach()
      list(APPEND CMAKE_REQUIRED_LIBRARIES "${REQUIRED_LIBS}")
      string(REGEX REPLACE "^ -" "-" CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}")

      # test link
      unset(LAPACKE_WORKS CACHE)
      include(CheckFunctionExists)
      check_function_exists(LAPACKE_dgeqrf LAPACKE_WORKS)
      mark_as_advanced(LAPACKE_WORKS)

      if(LAPACKE_WORKS)
        # save link with dependencies
        set(LAPACKE_LIBRARIES_DEP "${REQUIRED_LIBS}")
        set(LAPACKE_LIBRARY_DIRS_DEP "${REQUIRED_LIBDIRS}")
        set(LAPACKE_INCLUDE_DIRS_DEP "${REQUIRED_INCDIRS}")
        set(LAPACKE_LINKER_FLAGS "${REQUIRED_LDFLAGS}")
        list(REMOVE_DUPLICATES LAPACKE_LIBRARY_DIRS_DEP)
        list(REMOVE_DUPLICATES LAPACKE_INCLUDE_DIRS_DEP)
        list(REMOVE_DUPLICATES LAPACKE_LINKER_FLAGS)
      else()
        if(NOT LAPACKE_FIND_QUIETLY)
          message(STATUS "Looking for lapacke: test of LAPACKE_dgeqrf with lapacke and lapack libraries fails")
          message(STATUS "CMAKE_REQUIRED_LIBRARIES: ${CMAKE_REQUIRED_LIBRARIES}")
          message(STATUS "CMAKE_REQUIRED_INCLUDES: ${CMAKE_REQUIRED_INCLUDES}")
          message(STATUS "Check in CMakeFiles/CMakeError.log to figure out why it fails")
        endif()
      endif()
      set(CMAKE_REQUIRED_INCLUDES)
      set(CMAKE_REQUIRED_FLAGS)
      set(CMAKE_REQUIRED_LIBRARIES)
    endif(LAPACKE_LIBRARIES)

  endif (LAPACKE_STANDALONE OR NOT LAPACKE_WORKS)

else(LAPACK_FOUND)

  if (NOT LAPACKE_FIND_QUIETLY)
    message(STATUS "LAPACKE requires LAPACK but LAPACK has not been found."
      "Please look for LAPACK first.")
  endif()

endif(LAPACK_FOUND)

if (LAPACKE_LIBRARIES)
  list(GET LAPACKE_LIBRARIES 0 first_lib)
  get_filename_component(first_lib_path "${first_lib}" PATH)
  if (${first_lib_path} MATCHES "(/lib(32|64)?$)|(/lib/intel64$|/lib/ia32$)")
    string(REGEX REPLACE "(/lib(32|64)?$)|(/lib/intel64$|/lib/ia32$)" "" not_cached_dir "${first_lib_path}")
    set(LAPACKE_DIR_FOUND "${not_cached_dir}" CACHE PATH "Installation directory of LAPACKE library" FORCE)
  else()
    set(LAPACKE_DIR_FOUND "${first_lib_path}" CACHE PATH "Installation directory of LAPACKE library" FORCE)
  endif()
endif()
mark_as_advanced(LAPACKE_DIR)
mark_as_advanced(LAPACKE_DIR_FOUND)

# check that LAPACKE has been found
# ---------------------------------
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACKE DEFAULT_MSG
  LAPACKE_LIBRARIES
  LAPACKE_WORKS)
