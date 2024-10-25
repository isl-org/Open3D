# Find the Paddle root and use the provided cmake module
# The following variables will be set:
# - Paddle_FOUND
# - Paddle_VERSION
# - Paddle_ROOT
# - Paddle_DEFINITIONS
#
# - PADDLE_FOUND
# - PADDLE_INCLUDE_DIRS
# - PADDLE_LIBRARY_DIRS
# - PADDLE_LIBRARIES
# - PADDLE_CXX_FLAGS
#
# and import the target 'paddle'.

if(NOT Paddle_FOUND)
    # Searching for Paddle requires the python executable
    if (NOT Python3_EXECUTABLE)
        message(FATAL_ERROR "Python 3 not found in top level file")
    endif()

    if(BUILD_CUDA_MODULE)
        find_package(CUDAToolkit REQUIRED)
        string(SUBSTRING ${CUDAToolkit_VERSION} 0 4 CUDA_VERSION)
    endif()

    message(STATUS "Getting Paddle properties ...")

    set(Paddle_FETCH_PROPERTIES
        "import os"
        "import paddle"
        "import sysconfig"
        "print(paddle.__version__, end=';')"
        "print(os.path.dirname(paddle.__file__), end=';')"
        "print(sysconfig.get_path('include', scheme='posix_prefix'), end=';')"
    )
    execute_process(
        COMMAND ${Python3_EXECUTABLE} "-c" "${Paddle_FETCH_PROPERTIES}"
        OUTPUT_VARIABLE Paddle_PROPERTIES
    )


    list(GET Paddle_PROPERTIES 0 Paddle_VERSION)
    list(GET Paddle_PROPERTIES 1 Paddle_ROOT)
    list(GET Paddle_PROPERTIES 2 Python_INCLUDE)

    set(Paddle_CXX11_ABI True)

    unset(Paddle_FETCH_PROPERTIES)
    unset(Paddle_PROPERTIES)

    add_library(paddle STATIC IMPORTED)

    # handle include directories
    set(PADDLE_INCLUDE_DIRS)
    list(APPEND PADDLE_INCLUDE_DIRS "${Paddle_ROOT}/include")
    list(APPEND PADDLE_INCLUDE_DIRS "${Paddle_ROOT}/include/third_party")
    list(APPEND PADDLE_INCLUDE_DIRS "${Python_INCLUDE}")

    if(BUILD_CUDA_MODULE)
        list(APPEND PADDLE_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
    endif()

    # handle library directories
    set(PADDLE_LIBRARY_DIRS)
    list(APPEND PADDLE_LIBRARY_DIRS "${Paddle_ROOT}/libs")
    list(APPEND PADDLE_LIBRARY_DIRS "${Paddle_ROOT}/base")

    if(BUILD_CUDA_MODULE)
        list(APPEND PADDLE_LIBRARY_DIRS "${CUDAToolkit_LIBRARY_DIR}")
    endif()

    # handle libraries
    set(PADDLE_LIBRARIES)
    find_library(PADDLE_LIB NAMES paddle PATHS "${Paddle_ROOT}/base")
    list(APPEND PADDLE_LIBRARY_DIRS "${PADDLE_LIB}")

    if(BUILD_CUDA_MODULE)
        find_library(CUDART_LIB NAMES cudart PATHS "${CUDAToolkit_LIBRARY_DIR}")
        list(APPEND PADDLE_LIBRARY_DIRS "${CUDART_LIB}")
    endif() 

    # handle compile flags
    set(PADDLE_CXX_FLAGS)
    if(BUILD_CUDA_MODULE)
        set(PADDLE_CXX_FLAGS "-DPADDLE_WITH_CUDA ${PADDLE_CXX_FLAGS}")
    endif() 

    set_target_properties(paddle PROPERTIES
        IMPORTED_LOCATION "${PADDLE_LIB}"
    )
    set_target_properties(paddle PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${PADDLE_INCLUDE_DIRS}"
    )
    set_property(TARGET paddle PROPERTY INTERFACE_COMPILE_OPTIONS "${PADDLE_CXX_FLAGS}")

    set(PADDLE_FOUND True)
endif()

if(PRINT_ONCE)
    message(STATUS "Paddle         version: ${Paddle_VERSION}")
    message(STATUS "               root dir: ${Paddle_ROOT}")
    message(STATUS "               compile flags: ${PADDLE_CXX_FLAGS}")
    if (UNIX AND NOT APPLE)
        message(STATUS "               use cxx11 abi: ${Paddle_CXX11_ABI}")
    endif()
    foreach(idir ${PADDLE_INCLUDE_DIRS})
        message(STATUS "               include dirs: ${idir}")
    endforeach(idir)
    foreach(ldir ${PADDLE_LIBRARY_DIRS})
        message(STATUS "               library dirs: ${ldir}")
    endforeach(ldir)
    foreach(lib ${PADDLE_LIBRARIES})
        message(STATUS "               libraries: ${lib}")
    endforeach(lib)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Paddle DEFAULT_MSG Paddle_VERSION
                                  Paddle_ROOT)
