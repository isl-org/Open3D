cmake_minimum_required(VERSION 3.12.0)
# CMake 3.12+ is required to allow linking with OBJECT libraries
# and to prevent erroneous -gencode option deduplication with CUDA.
# If you're using Ubuntu 18.04 or older, we suggest you install
# a backported CMake from https://apt.kitware.com/

set (CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum OS X deployment version")

# Config options
option(BUILD_SHARED_LIBS         "Build shared libraries"                   OFF)
option(WITH_OPENMP               "Use OpenMP multi-threading"               ON)
option(ENABLE_HEADLESS_RENDERING "Use OSMesa for headless rendering"        OFF)
option(BUILD_CPP_EXAMPLES        "Build the Open3D example programs"        ON)
option(BUILD_UNIT_TESTS          "Build the Open3D unit tests"              OFF)
option(BUILD_GOOGLETEST          "Build googletest framework from source"   ON)
option(BUILD_BENCHMARKS          "Build the micro benchmarks"               OFF)
option(BUILD_EIGEN3              "Use the Eigen3 that comes with Open3D"    ON)
option(BUILD_FILAMENT            "Build filament from source"               OFF)
option(BUILD_FLANN               "Use the flann that comes with Open3D"     ON)
option(BUILD_FMT                 "Build fmtlib from source"                 ON)
option(BUILD_GLEW                "Build glew from source"                   OFF)
option(BUILD_GLFW                "Build glfw from source"                   OFF)
option(BUILD_IMGUI               "Build Dear ImGui from source"             ON)
option(BUILD_LIBLZF              "Build liblzf from source"                 ON)
option(BUILD_PNG                 "Build png from source"                    OFF)
option(BUILD_ZLIB                "Build zlib from source"                   OFF)
option(BUILD_JPEG                "Build jpeg-turbo from source"             OFF)
option(BUILD_PYBIND11            "Build pybind11 from source"               ON)
option(BUILD_PYTHON_MODULE       "Build the python module"                  ON)
option(BUILD_LIBREALSENSE        "Build support for Intel RealSense camera" OFF)
option(BUILD_AZURE_KINECT        "Build support for Azure Kinect sensor"    OFF)
option(BUILD_TINYFILEDIALOGS     "Build tinyfiledialogs from source"        ON)
option(BUILD_TINYGLTF            "Build TinyGLTF from source"               ON)
option(BUILD_TINYOBJLOADER       "Build tinyobjloader from source"          ON)
option(BUILD_QHULL               "Build qhull from source"                  ON)
option(BUILD_CUDA_MODULE         "Build the CUDA module"                    OFF)
option(BUILD_TENSORFLOW_OPS      "Build ops for Tensorflow"                 OFF)
option(BUILD_PYTORCH_OPS         "Build ops for Pytorch"                    OFF)
option(ENABLE_GUI                "Builds new GUI"                           ON)
option(ENABLE_JUPYTER            "Enable Jupyter support for Open3D"        ON)
option(USE_VULKAN                "Enables Vulkan support for renderer"      OFF)
option(STATIC_WINDOWS_RUNTIME    "Use static (MT/MTd) Windows runtime"      OFF)
option(GLIBCXX_USE_CXX11_ABI     "Set -D_GLIBCXX_USE_CXX11_ABI=1"           OFF)
option(PREFER_OSX_HOMEBREW       "Prefer Homebrew libs over frameworks"     ON)

set(FILAMENT_PRECOMPILED_ROOT "" CACHE PATH "Path to precompiled Filament library (used if BUILD_FILAMENT=OFF)")
set(FILAMENT_SOURCE_ROOT "" CACHE PATH "Path to Filament library sources (used if BUILD_FILAMENT=ON)")

if (PREFER_OSX_HOMEBREW)
    set(CMAKE_FIND_FRAMEWORK LAST)
    set(CMAKE_FIND_APPBUNDLE LAST)
endif()

# Cache variables for specifying the GPU architectures
set(CUDA_ARCH "Auto" CACHE STRING "Selects GPU architectures for code generation, \
one of (Auto|BasicPTX|User). Set to 'User' to set a custom list of architectures" )
set_property(CACHE CUDA_ARCH PROPERTY STRINGS Auto BasicPTX User)
set(CUDA_ARCH_USER "" CACHE STRING "User defined list of GPU architectures, e.g. 3.5 5.0+PTX Turing" )

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No CMAKE_BUILD_TYPE specified, default to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

# Set OpenGL policy
if (POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
    set(CMP0072_VALUE NEW)
    if (BUILD_GLFW)
        cmake_policy(SET CMP0072 OLD)
        set(CMP0072_VALUE OLD)
    endif()
endif()
# Set policy for hidden symbol visibility
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()
# Set policy to enable MSVC_RUNTIME_LIBRARY property
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

# Catch a few incompatible build options
if((UNIX AND NOT APPLE) AND BUILD_LIBREALSENSE)
    message(STATUS "Disabling RealSense since it is not fully supported on Linux.")
    set(BUILD_LIBREALSENSE OFF)
endif()
if (APPLE AND ENABLE_HEADLESS_RENDERING)
    message(STATUS "Headless rendering is not supported on Mac OS")
    set(ENABLE_HEADLESS_RENDERING OFF)
endif()
if(ENABLE_HEADLESS_RENDERING AND ENABLE_GUI)
    message(STATUS "Headless rendering disables the Open3D GUI")
    set(ENABLE_GUI OFF)
endif()
if(ENABLE_HEADLESS_RENDERING AND (NOT BUILD_GLEW OR NOT BUILD_GLFW))
    message(STATUS "Headless rendering requires customized GLEW and GLFW builds")
    set(BUILD_GLEW ON)
    set(BUILD_GLFW ON)
endif()
if(ENABLE_GUI AND WIN32)
    message(STATUS "New GUI is currently unsupported on Windows")
    set(ENABLE_GUI OFF)
endif()

# Parse Open3D version number
file(STRINGS "cpp/open3d/version.txt" OPEN3D_VERSION_READ)
foreach(ver ${OPEN3D_VERSION_READ})
    if (ver MATCHES "OPEN3D_VERSION_(MAJOR|MINOR|PATCH|TWEAK) +([^ ]+)$")
        set(OPEN3D_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
string(CONCAT OPEN3D_VERSION
    "${OPEN3D_VERSION_MAJOR}"
    ".${OPEN3D_VERSION_MINOR}"
    ".${OPEN3D_VERSION_PATCH}"
    ".${OPEN3D_VERSION_TWEAK}"
)

project(Open3D VERSION ${OPEN3D_VERSION} LANGUAGES C CXX)
message(STATUS "Open3D ${PROJECT_VERSION}")

# CMake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/3rdparty/CMake)

# setup PYTHON_EXECUTABLE if not set
find_package(PythonExecutable REQUIRED) # invokes the module in 3rdparty/CMake

# npm version has to be MAJOR.MINOR.PATCH
string(CONCAT PROJECT_VERSION_THREE_NUMBER "${OPEN3D_VERSION_MAJOR}"
                                           ".${OPEN3D_VERSION_MINOR}"
                                           ".${OPEN3D_VERSION_PATCH}")

# PyPI package name controls specifies the repository name on PyPI. The default
# name is "open3d". In the past, for historical reasons, we've used the
# following names for PyPI, while they are now deprecated:
# - open3d-python
# - py3d
# - open3d-original
# - open3d-official
# - open-3d
if(NOT DEFINED PYPI_PACKAGE_NAME)
    set(PYPI_PACKAGE_NAME "open3d")
endif()

# Set additional info
set(PROJECT_EMAIL   "info@open3d.org")
set(PROJECT_HOME    "http://www.open3d.org")
set(PROJECT_DOCS    "http://www.open3d.org/docs")
set(PROJECT_CODE    "https://github.com/intel-isl/Open3D")
set(PROJECT_ISSUES  "https://github.com/intel-isl/Open3D/issues")

# Set installation paths
if(UNIX OR CYGWIN)
    include(GNUInstallDirs)
    set(Open3D_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
    set(Open3D_INSTALL_BIN_DIR "${CMAKE_INSTALL_BINDIR}")
    set(Open3D_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(Open3D_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
else()
    set(Open3D_INSTALL_INCLUDE_DIR include)
    set(Open3D_INSTALL_BIN_DIR bin)
    set(Open3D_INSTALL_LIB_DIR lib)
    set(Open3D_INSTALL_CMAKE_DIR CMake)
endif()

# Put build results in some predictable places
# The $<CONFIG> generator expression makes sure that XCode or Visual Studio do not
# append additional path components, as we need to know *exactly* where the build results
# end up.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/$<CONFIG>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Build CUDA module by default if CUDA is available
# Compatible with CMake 3.8+
if (BUILD_CUDA_MODULE)
    include(CheckLanguage)
    check_language(CUDA)
    message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    if(CMAKE_CUDA_COMPILER)
        find_package(CUDA REQUIRED) # required for cuda_select_nvcc_arch_flags
        message(STATUS "Building CUDA enabled")
        enable_language(CUDA)
        # Get gencode flags
        if("${CUDA_ARCH}" STREQUAL "User")
            cuda_select_nvcc_arch_flags(CUDA_GENCODES "${CUDA_ARCH_USER}")
        elseif("${CUDA_ARCH}" STREQUAL "BasicPTX")
            # Include oldest and most recent PTX and rely on JIT compilation
            set(CUDA_GENCODES "-gencode arch=compute_30,code=compute_30;-gencode arch=compute_75,code=compute_75")
        else()
            cuda_select_nvcc_arch_flags(CUDA_GENCODES "${CUDA_ARCH}")
        endif()
        # Make CUDA_GENCODES a string to avoid deduplication in target_compile_options
        string( REPLACE ";" " " CUDA_GENCODES "${CUDA_GENCODES}")
        message(STATUS "CUDA_GENCODES: ${CUDA_GENCODES}")
        # Allows CUDA headers to be included as "system" headers for CMake prior
        # to 3.17,
        # https://gitlab.kitware.com/cmake/cmake/-/issues/20270
        # https://github.com/intel-isl/Open3D/issues/2040
        if(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
            list(REMOVE_ITEM CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES
                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        endif()
        message(STATUS "CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
        message(STATUS "CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}")
    else()
        set(BUILD_CUDA_MODULE OFF)
        message(STATUS "No CUDA support")
    endif()
endif ()

# OS specific settings
if(ENABLE_GUI AND WIN32)
    # Filament links normally only against static runtime
    set(STATIC_WINDOWS_RUNTIME ON)
    if(NOT COMMAND target_link_options)  # CMake 3.13+ required
        # GLEW and Open3D make direct OpenGL calls and link to opengl32.lib;
        # Filament needs to link through bluegl.lib.
        # See https://github.com/google/filament/issues/652
        string(APPEND CMAKE_EXE_LINKER_FLAGS " /force:multiple")
        string(APPEND CMAKE_SHARED_LINKER_FLAGS " /force:multiple")
    endif()
endif()
if(WIN32)
    # Windows defaults to hidden symbol visibility, override that
    # TODO: It would be better to explictly export symbols.
    #       Then, we could use -fvisibility=hidden for Linux as well
    SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if(MSVC)
        # Make sure we don't hit the 65535 object member limit with MSVC
        #
        # /bigobj allows object files with more than 65535 members
        # /Ob2 enables function inlining, because MSVC is particularly
        # verbose with inline members
        #
        # See: https://github.com/tensorflow/tensorflow/pull/10962
        add_compile_options(/bigobj /Ob2)
    endif()
    if (STATIC_WINDOWS_RUNTIME)
        # This needs cmake_policy(SET CMP0091 NEW)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

# Folder view for project files
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Convenience function to link against all third-party libraries
# We need this because we create a lot of object libraries to assemble
# the main Open3D library
function(open3d_link_3rdparty_libraries target)
    target_link_libraries(${target} PRIVATE ${Open3D_3RDPARTY_PRIVATE_TARGETS})
    target_link_libraries(${target} PUBLIC ${Open3D_3RDPARTY_PUBLIC_TARGETS})
    foreach(dep IN LISTS Open3D_3RDPARTY_HEADER_TARGETS)
        if(TARGET ${dep})
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
            if(inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${inc})
            endif()
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
            if(inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${inc})
            endif()
            get_property(def TARGET ${dep} PROPERTY INTERFACE_COMPILE_DEFINITIONS)
            if(def)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ${def})
            endif()
        endif()
    endforeach()
endfunction()

# Check if the compiler defines the _GLIBCXX_USE_CXX11_ABI macro
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("#include <cxxabi.h>
int main() { return _GLIBCXX_USE_CXX11_ABI; }" HAS_GLIBCXX_USE_CXX11_ABI)

# Convenience function to set important target properties
function(open3d_set_global_properties target)
    # Libraries need to be compiled with position independent code
    get_target_property(target_type ${target} TYPE)
    if (NOT target_type STREQUAL "EXECUTABLE")
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
    # Tell CMake we want a compiler that supports C++14 features
    target_compile_features(${target} PUBLIC cxx_variadic_templates cxx_constexpr cxx_override cxx_static_assert cxx_trailing_return_types cxx_return_type_deduction)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp>
        $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}>
    )
    if(BUILD_CUDA_MODULE)
        target_compile_definitions(${target} PRIVATE BUILD_CUDA_MODULE)
    endif()
    if(GLIBCXX_USE_CXX11_ABI)
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
    else()
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
    endif()
    if(ENABLE_HEADLESS_RENDERING)
        target_compile_definitions(${target} PRIVATE HEADLESS_RENDERING)
    endif()
    if(WIN32)
        target_compile_definitions(${target} PRIVATE
            WINDOWS
            _CRT_SECURE_NO_DEPRECATE
            _CRT_NONSTDC_NO_DEPRECATE
            _SCL_SECURE_NO_WARNINGS
        )
        if(MSVC)
            target_compile_definitions(${target} PRIVATE NOMINMAX _USE_MATH_DEFINES _ENABLE_EXTENDED_ALIGNED_STORAGE)
            target_compile_options(${target} PRIVATE /EHsc /wd4522 /Wv:18 /WX)
            # Multi-thread compile, two ways to enable
            # Option 1, at build time: cmake --build . --parallel %NUMBER_OF_PROCESSORS%
            # https://stackoverflow.com/questions/36633074/set-the-number-of-threads-in-a-cmake-build
            # OPtion 2, at configure time: add /MP flag, no need to use Option 1
            # https://docs.microsoft.com/en-us/cpp/build/reference/mp-build-with-multiple-processes?view=vs-2019
            #
            # /bigobj is enabled by default. On Windows, the build type (Debug, Release)
            # is set at build time, not config time, since CMAKE_BUILD_TYPE is not
            # supported on Windows.
            # ref: https://stackoverflow.com/a/15110915
            # ref: https://stackoverflow.com/a/19026241
            target_compile_options(${target} PRIVATE /MP)
            if(COMMAND target_link_options)
                if(ENABLE_GUI)
                    # GLEW and Open3D make direct OpenGL calls and link to opengl32.lib;
                    # Filament needs to link through bluegl.lib.
                    # See https://github.com/google/filament/issues/652
                    target_link_options(${target} PRIVATE /force:multiple)
                endif()
            endif()
        endif()
    elseif(APPLE)
        target_compile_definitions(${target} PRIVATE UNIX APPLE)
        target_compile_options(${target} PRIVATE "-Wno-deprecated-declarations")
    elseif(UNIX)
        target_compile_definitions(${target} PRIVATE UNIX)
        target_compile_options(${target} PRIVATE "-Wno-deprecated-declarations" "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-result>")
    endif()
    # The SHELL: prefix requires CMake 3.12+
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:${CUDA_GENCODES}>")
endfunction()

macro(add_source_group module_name)
    file(GLOB MODULE_HEADER_FILES "${module_name}/*.h")
    source_group("Header Files\\${module_name}" FILES ${MODULE_HEADER_FILES})
    file(GLOB MODULE_SOURCE_FILES "${module_name}/*.cpp")
    source_group("Source Files\\${module_name}" FILES ${MODULE_SOURCE_FILES})
endmacro()

# Enumerate all third-party libraries which we need later
# This creates the necessary targets and sets the
# Open3D_3RDPARTY_*_TARGETS variables we use in open3d_link_3rdparty_libraries
include(3rdparty/find_dependencies.cmake)

# Open3D library
add_subdirectory(cpp)

# Examples
add_subdirectory(examples)

# Install CMake configuration files
install(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}:: DESTINATION ${Open3D_INSTALL_CMAKE_DIR})
export(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}::)

# `make check-style` checks style for c++/cuda/python/ipynb files
add_custom_target(check-style
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/util/scripts/check_style.py
    COMMENT "Python executable used for style check: ${PYTHON_EXECUTABLE}."
)

# `make apply-style` applies style for c++/cuda/python/ipynb files
add_custom_target(apply-style
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/util/scripts/check_style.py --do_apply_style
    COMMENT "Python executable used for style check: ${PYTHON_EXECUTABLE}."
)

# `make check-cpp-style` checks style for c++/cuda files.
# This works outside of python virtualenv.
add_custom_target(check-cpp-style
    COMMAND ${CMAKE_COMMAND}
    -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    -DDO_APPLY_STYLE=OFF
    -P ${CMAKE_CURRENT_SOURCE_DIR}/util/scripts/check_cpp_style.cmake
)

# `make apply-cpp-style` applies style for c++/cuda files.
# This works outside of python virtualenv.
add_custom_target(apply-cpp-style
    COMMAND ${CMAKE_COMMAND}
    -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    -DDO_APPLY_STYLE=ON
    -P ${CMAKE_CURRENT_SOURCE_DIR}/util/scripts/check_cpp_style.cmake
)

function(open3d_aligned_print first second)
    string(LENGTH "${first}" first_len)
    math(EXPR star_len "40 - ${first_len}")
    # CMake 3.15+ has string(REPEAT), but this is backwards compatible
    string(SUBSTRING "........................................" 0 ${star_len} stars)
    message(STATUS "  ${first} ${stars} ${second}")
endfunction()

message(STATUS "========================================================================")
message(STATUS "Open3D ${PROJECT_VERSION} Configuration Summary")
message(STATUS "========================================================================")
message(STATUS "Enabled Features:")
open3d_aligned_print("OpenMP" "${WITH_OPENMP}")
open3d_aligned_print("Headless Rendering" "${ENABLE_HEADLESS_RENDERING}")
open3d_aligned_print("Azure Kinect Support" "${BUILD_AZURE_KINECT}")
open3d_aligned_print("CUDA Support" "${BUILD_CUDA_MODULE}")
open3d_aligned_print("Build GUI" "${ENABLE_GUI}")
open3d_aligned_print("Build Shared Library" "${BUILD_SHARED_LIBS}")
open3d_aligned_print("Build Unit Tests" "${BUILD_UNIT_TESTS}")
open3d_aligned_print("Build Examples" "${BUILD_CPP_EXAMPLES}")
open3d_aligned_print("Build Python Module" "${BUILD_PYTHON_MODULE}")
if(BUILD_PYTHON_MODULE)
    open3d_aligned_print("- with Jupyter Notebook Support" "${ENABLE_JUPYTER}")
endif()
open3d_aligned_print("Build Tensorflow Ops" "${BUILD_TENSORFLOW_OPS}")
open3d_aligned_print("Build Pytorch Ops" "${BUILD_PYTORCH_OPS}")
open3d_aligned_print("Build Benchmarks" "${BUILD_BENCHMARKS}")
if(GLIBCXX_USE_CXX11_ABI)
    set(usage "1")
else()
    set(usage "0")
endif()
open3d_aligned_print("Force GLIBCXX_USE_CXX11_ABI=" "${usage}")
message(STATUS "========================================================================")
message(STATUS "Third-Party Dependencies:")
foreach(dep IN ITEMS EIGEN3 FILAMENT FLANN FMT GLEW GLFW GOOGLETEST IMGUI JPEG JSONCPP LIBLZF OPENGL PNG PYBIND11 QHULL LIBREALSENSE TINYFILEDIALOGS TINYGLTF TINYOBJLOADER)
    if(${dep}_TARGET)
        if(BUILD_${dep})
            set(usage "yes (build from source)")
        else()
            set(usage "yes")
            if(${dep}_VERSION_STRING)
                set(usage "${usage} (v${${dep}_VERSION_STRING})")
            elseif(${dep}_VERSION)
                set(usage "${usage} (v${${dep}_VERSION})")
            endif()
        endif()
    else()
        set(usage "no")
    endif()
    open3d_aligned_print("${dep}" "${usage}")
endforeach()
message(STATUS "========================================================================")
