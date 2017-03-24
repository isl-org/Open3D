project(libpng C)

cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

set(PNGLIB_MAJOR 1)
set(PNGLIB_MINOR 6)
set(PNGLIB_RELEASE 18)
set(PNGLIB_VERSION ${PNGLIB_MAJOR}.${PNGLIB_MINOR}.${PNGLIB_RELEASE})

# needed packages
include_directories("${Open3D_SOURCE_DIR}/External/zlib")

if(NOT WIN32)
  find_library(M_LIBRARY
    NAMES m
    PATHS /usr/lib /usr/local/lib
  )
  if(NOT M_LIBRARY)
    message(STATUS "math lib 'libm' not found; floating point support disabled")
  endif()
else()
  # not needed on windows
  set(M_LIBRARY "")
endif()

# SET LIBNAME
set(PNG_LIBRARY png)

# Use the prebuilt pnglibconf.h file from the scripts folder
# TODO: fix this by building with awk; without this no cmake build can be
# configured directly (to do so indirectly use your local awk to build a
# pnglibconf.h in the build directory.)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pnglibconf.h.prebuilt
#               ${CMAKE_CURRENT_BINARY_DIR}/pnglibconf.h)
#include_directories(${CMAKE_CURRENT_BINARY_DIR})

# OUR SOURCES
set(libpng_public_hdrs
  png.h
  pngconf.h
#  ${CMAKE_CURRENT_BINARY_DIR}/pnglibconf.h
  pnglibconf.h
)
set(libpng_sources
  ${libpng_public_hdrs}
  pngdebug.h
  pnginfo.h
  pngpriv.h
  pngstruct.h
  png.c
  pngerror.c
  pngget.c
  pngmem.c
  pngpread.c
  pngread.c
  pngrio.c
  pngrtran.c
  pngrutil.c
  pngset.c
  pngtrans.c
  pngwio.c
  pngwrite.c
  pngwtran.c
  pngwutil.c
)
# SOME NEEDED DEFINITIONS

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
endif(MSVC)

add_library(${PNG_LIBRARY} STATIC ${libpng_sources})
target_link_libraries(${PNG_LIBRARY} ${ZLIB_LIBRARY} ${M_LIBRARY})
set_target_properties(${PNG_LIBRARY} PROPERTIES
        OUTPUT_NAME ${PNG_LIBRARY}
        FOLDER "External"
    )

set(PNG_INCLUDE_DIRS "${Open3D_SOURCE_DIR}/External/libpng")
set(PNG_LIBRARIES "png")
set(PNG_INCLUDE_DIRS ${PNG_INCLUDE_DIRS} PARENT_SCOPE)
set(PNG_LIBRARIES ${PNG_LIBRARIES} PARENT_SCOPE)
