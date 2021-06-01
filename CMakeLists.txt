cmake_minimum_required(VERSION 3.18)
# If you're using Ubuntu 18.04, we suggest you install the latest CMake from the
# official repository https://apt.kitware.com/.  CMake 3.18+ is required to
# allow linking with OBJECT libraries, to prevent erroneous -gencode option
# deduplication with CUDA, and to simplify generator expressions for selecting
# compile flags and setting global hardened link flags.

if (APPLE)
set (CMAKE_OSX_DEPLOYMENT_TARGET "10.14" CACHE STRING
    "Minimum OS X deployment version" FORCE)
endif()

if(UNIX AND NOT APPLE)
    # CMAKE_HOST_SYSTEM_PROCESSOR is only available after calling project(),
    # which depends on ${OPEN3D_VERSION}, which depends on ${DEVELOPER_BUILD}.
    execute_process(COMMAND uname -p
        OUTPUT_VARIABLE PROCESSOR_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PROCESSOR_ARCH STREQUAL "aarch64")
        set(LINUX_AARCH64 TRUE)
    endif()
endif()

include(CMakeDependentOption)

# Open3D build options
option(BUILD_SHARED_LIBS          "Build shared libraries"                   OFF)
option(BUILD_EXAMPLES             "Build Open3D examples programs"           ON )
option(BUILD_UNIT_TESTS           "Build Open3D unit tests"                  OFF)
option(BUILD_BENCHMARKS           "Build the micro benchmarks"               OFF)
option(BUILD_PYTHON_MODULE        "Build the python module"                  ON )
option(BUILD_CUDA_MODULE          "Build the CUDA module"                    OFF)
option(BUILD_COMMON_CUDA_ARCHS    "Build for common CUDA GPUs (for release)" OFF)
option(BUILD_CACHED_CUDA_MANAGER  "Build the cached CUDA memory manager"     ON )
option(BUILD_GUI                  "Builds new GUI"                           ON )
option(WITH_OPENMP                "Use OpenMP multi-threading"               ON )
option(WITH_IPPICV                "Use Intel Performance Primitives"         ON )
option(ENABLE_HEADLESS_RENDERING  "Use OSMesa for headless rendering"        OFF)
cmake_dependent_option(
       STATIC_WINDOWS_RUNTIME     "Use static (MT/MTd) Windows runtime"      ON
       "NOT BUILD_SHARED_LIBS"                                               OFF)
option(GLIBCXX_USE_CXX11_ABI      "Set -D_GLIBCXX_USE_CXX11_ABI=1"           OFF)
option(BUILD_RPC_INTERFACE        "Build the RPC interface"                  OFF)
if(WIN32 OR UNIX AND NOT LINUX_AARCH64)
    cmake_dependent_option(BUILD_WEBRTC "Build WebRTC visualizer" ON "BUILD_GUI" OFF)
else()
    option(BUILD_WEBRTC           "Build WebRTC visualizer"                  OFF)
endif()
option(BUILD_JUPYTER_EXTENSION    "Build Jupyter, requires BUILD_WEBRTC=ON"  OFF)

# 3rd-party build options
option(USE_BLAS                   "Use BLAS/LAPACK instead of MKL"           OFF)
option(USE_SYSTEM_EIGEN3          "Use system pre-installed eigen3"          OFF)
option(USE_SYSTEM_FLANN           "Use system pre-installed flann"           OFF)
option(USE_SYSTEM_FMT             "Use system pre-installed fmt"             OFF)
option(USE_SYSTEM_GLEW            "Use system pre-installed glew"            OFF)
option(USE_SYSTEM_GLFW            "Use system pre-installed glfw"            OFF)
option(USE_SYSTEM_GOOGLETEST      "Use system pre-installed googletest"      OFF)
option(USE_SYSTEM_IMGUI           "Use system pre-installed imgui"           OFF)
option(USE_SYSTEM_JPEG            "Use system pre-installed jpeg"            OFF)
option(USE_SYSTEM_LIBLZF          "Use system pre-installed liblzf"          OFF)
option(USE_SYSTEM_PNG             "Use system pre-installed png"             OFF)
option(USE_SYSTEM_PYBIND11        "Use system pre-installed pybind11"        OFF)
option(USE_SYSTEM_QHULL           "Use system pre-installed qhull"           OFF)
option(USE_SYSTEM_TINYGLTF        "Use system pre-installed tinygltf"        OFF)
option(USE_SYSTEM_TINYOBJLOADER   "Use system pre-installed tinyobjloader"   OFF)
option(BUILD_FILAMENT_FROM_SOURCE "Build filament from source"               OFF)
option(PREFER_OSX_HOMEBREW        "Prefer Homebrew libs over frameworks"     ON )
option(WITH_FAISS                 "Enable Faiss"                             ON )

# Sensor options
option(BUILD_LIBREALSENSE         "Build support for Intel RealSense camera" OFF)
option(USE_SYSTEM_LIBREALSENSE    "Use system pre-installed librealsense"    OFF)
option(BUILD_AZURE_KINECT         "Build support for Azure Kinect sensor"    OFF)

# ML library options
option(BUILD_TENSORFLOW_OPS       "Build ops for TensorFlow"                 OFF)
option(BUILD_PYTORCH_OPS          "Build ops for PyTorch"                    OFF)
option(BUNDLE_OPEN3D_ML           "Includes the Open3D-ML repo in the wheel" OFF)

# Release build options
option(DEVELOPER_BUILD      "Add +commit_hash to the project version number" ON )
if (NOT DEVELOPER_BUILD)
    if (NOT BUILD_COMMON_CUDA_ARCHS)
        set(BUILD_COMMON_CUDA_ARCHS ON CACHE BOOL "Build for common CUDA GPUs (for release)" FORCE)
        message(WARNING "Setting BUILD_COMMON_CUDA_ARCHS=ON since DEVELOPER_BUILD is OFF.")
    endif()
endif()

# In ExternalProject_Add, if OPEN3D_THIRD_PARTY_DOWNLOAD_DIR is specified, CMake will
# first try to look for the required files in OPEN3D_THIRD_PARTY_DOWNLOAD_DIR, before
# downloading it from the internet. The files inside OPEN3D_THIRD_PARTY_DOWNLOAD_DIR
# are prepared manually by the user. This is only supported by limited 3rd party
# libraries.

# Default build type on single-config generators.
# For multi-config generators (e.g. Visual Studio), CMAKE_CONFIGURATION_TYPES
# will be set, and we don't specify a default CMAKE_BUILD_TYPE.
# https://blog.kitware.com/cmake-and-the-default-build-type/
if(NOT CMAKE_CONFIGURATION_TYPES)
    if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "Setting build type to RelWithDebInfo as none was specified.")
        set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
        # Set the possible values of build type for cmake-gui.
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    endif()
    message(STATUS "CMAKE_BUILD_TYPE is set to ${CMAKE_BUILD_TYPE}.")
endif()

find_program(CCACHE "ccache")
if (CCACHE)
    message(STATUS "ccache found at ${CCACHE}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    if(BUILD_CUDA_MODULE)
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE})
    endif()
endif()

set(THIRD_PARTY_FROM "cmake options")
if(DEFINED ENV{OPEN3D_THIRD_PARTY_DOWNLOAD_DIR} AND
        IS_DIRECTORY "$ENV{OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}" AND
        NOT OPEN3D_THIRD_PARTY_DOWNLOAD_DIR)
    set(OPEN3D_THIRD_PARTY_DOWNLOAD_DIR "$ENV{OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}")
    set(THIRD_PARTY_FROM "the environment")
endif()
if(OPEN3D_THIRD_PARTY_DOWNLOAD_DIR)
    message(STATUS "OPEN3D_THIRD_PARTY_DOWNLOAD_DIR is set to "
    "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR} from ${THIRD_PARTY_FROM}.")
else()
    message(STATUS "OPEN3D_THIRD_PARTY_DOWNLOAD_DIR is not specified, will download directly.")
endif()

set(FILAMENT_PRECOMPILED_ROOT "" CACHE PATH "Path to precompiled Filament library (used if BUILD_FILAMENT_FROM_SOURCE=OFF)")

if (PREFER_OSX_HOMEBREW)
    set(CMAKE_FIND_FRAMEWORK LAST)
    set(CMAKE_FIND_APPBUNDLE LAST)
endif()

# Set OpenGL policy
if(NOT USE_SYSTEM_GLFW)
    cmake_policy(SET CMP0072 OLD)
endif()
cmake_policy(GET CMP0072 CMP0072_VALUE)

# Set BLAS/LAPACK library preference.
if(LINUX_AARCH64)
    if(NOT USE_BLAS)
        set(USE_BLAS ON)
        message(WARNING "ARM CPU detected, setting USE_BLAS=ON.")
    endif()
    if(BUILD_GUI AND (NOT BUILD_FILAMENT_FROM_SOURCE))
        set(BUILD_FILAMENT_FROM_SOURCE ON)
        message(WARNING "ARM CPU detected, setting BUILD_FILAMENT_FROM_SOURCE=ON.")
    endif()
endif()

# Catch a few incompatible build options
if (APPLE AND ENABLE_HEADLESS_RENDERING)
    message(WARNING "Headless rendering is not supported on Mac OS")
    set(ENABLE_HEADLESS_RENDERING OFF)
endif()
if(ENABLE_HEADLESS_RENDERING AND BUILD_GUI)
    message(WARNING "Headless rendering disables the Open3D GUI")
    set(BUILD_GUI OFF)
endif()
if(ENABLE_HEADLESS_RENDERING AND (USE_SYSTEM_GLEW OR USE_SYSTEM_GLFW))
    message(WARNING "Headless rendering requires customized GLEW and GLFW builds")
    set(USE_SYSTEM_GLEW OFF)
    set(USE_SYSTEM_GLFW OFF)
endif()
if(BUNDLE_OPEN3D_ML AND NOT (BUILD_TENSORFLOW_OPS OR BUILD_PYTORCH_OPS))
    message(SEND_ERROR "3DML depends on TensorFlow or PyTorch Ops. Enable them with -DBUILD_TENSORFLOW_OPS=ON or -DBUILD_PYTORCH_OPS=ON")
endif()
if(BUILD_WEBRTC AND LINUX_AARCH64)
    message(FATAL_ERROR "BUILD_WEBRTC=ON is not yet supported on ARM Linux")
endif()
if(BUILD_WEBRTC AND MSVC AND (BUILD_SHARED OR NOT STATIC_WINDOWS_RUNTIME))
    message(FATAL_ERROR "Please use STATIC_WINDOWS_RUNTIME=ON and "
    "BUILD_SHARED=OFF to build WebRTC with MSVC")
endif()
if(BUILD_WEBRTC AND NOT BUILD_GUI)
    message(FATAL_ERROR "BUILD_WEBRTC=ON requires BUILD_GUI=ON")
endif()
if(BUILD_JUPYTER_EXTENSION AND NOT BUILD_WEBRTC)
    # BUILD_JUPYTER_EXTENSION transitively depends on BUILD_GUI
    message(FATAL_ERROR "BUILD_JUPYTER_EXTENSION=ON requires BUILD_WEBRTC=ON")
endif()
if(BUILD_JUPYTER_EXTENSION AND NOT BUILD_PYTHON_MODULE)
    message(FATAL_ERROR "BUILD_JUPYTER_EXTENSION=ON requires BUILD_PYTHON_MODULE=ON")
endif()

# Parse Open3D version number
file(STRINGS "cpp/open3d/version.txt" OPEN3D_VERSION_READ)
foreach(ver ${OPEN3D_VERSION_READ})
    if (ver MATCHES "OPEN3D_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(OPEN3D_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(OPEN3D_VERSION_DEVHASH "")
if(DEVELOPER_BUILD)
    execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" log --pretty=format:%h -n 1
        OUTPUT_VARIABLE GIT_REV)
    if (GIT_REV)
        set(OPEN3D_VERSION_DEVHASH "+${GIT_REV}")
    endif()
endif()
string(CONCAT OPEN3D_VERSION
    "${OPEN3D_VERSION_MAJOR}"
    ".${OPEN3D_VERSION_MINOR}"
    ".${OPEN3D_VERSION_PATCH}"
)
set(OPEN3D_VERSION_FULL "${OPEN3D_VERSION}${OPEN3D_VERSION_DEVHASH}")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
    ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/getting_started.rst.in
    ${CMAKE_CURRENT_SOURCE_DIR}/docs/getting_started.rst @ONLY)

project(Open3D VERSION ${OPEN3D_VERSION} LANGUAGES C CXX)
message(STATUS "Open3D ${OPEN3D_VERSION_FULL}")

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

# Global Security options (including 3rd party code)
# Add -fPIC for library and -fPIE for executable to compiler and linker. Does not add -pie !
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# -Wall -Wextra -Werror or /W4 /WX are enabled for Open3D code (not 3rd party)
if (MSVC)
    set(HARDENING_CFLAGS
        /sdl            # SDL Checks
        /GS             # Code Generation: Security Check
        /guard:cf       # Code Generation: Control Flow Guard
        CACHE STRING "Compiler flags for security hardening")
    set(HARDENING_LDFLAGS
        /INCREMENTAL:NO  # Disable incremental Linking
        /NXCOMPAT        # Data Execution Prevention: On by default in VS2019
        /DYNAMICBASE     # Randomized Base Address
        /HIGHENTROPYVA   #
        #/INTEGRITYCHECK # Signed binary: Disabled
        CACHE STRING "Linker flags for security hardening")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_definitions(_FORTIFY_SOURCE=2)     # Buffer overflow detection
    set(HARDENING_CFLAGS
        -fstack-protector               # Stack-based buffer overrun detection
        -Wformat -Wformat-security      # Format string vulnerability
        CACHE STRING "Compiler flags for security hardening")
    set(HARDENING_LDFLAGS
        -fsanitize=safe-stack       # Stack execution protection
        -Wl,-z,relro,-z,now         # Data relocation protection
        -pie                        # Position independent executable
        CACHE STRING "Linker flags for security hardening")
    if(NOT BUILD_SHARED_LIBS AND NOT BUILD_PYTHON_MODULE)
        list(APPEND HARDENING_CFLAGS -fsanitize=safe-stack)   # Stack execution protection
        list(APPEND HARDENING_LDFLAGS -fsanitize=safe-stack)  # only static libraries supported
    endif()
    if(NOT DEVELOPER_BUILD)
        list(APPEND HARDENING_CFLAGS -O2)
        list(APPEND HARDENING_LDFLAGS -S)     # Strip debug symbols
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    add_compile_definitions(_FORTIFY_SOURCE=2)     # Buffer overflow detection
    set(HARDENING_CFLAGS
        -fstack-protector               # Stack-based buffer overrun detection
        -Wformat -Wformat-security      # Format string vulnerability
        CACHE STRING "Compiler flags for security hardening")
    set(HARDENING_LDFLAGS  ""
        # -pie Position independent execution is default on macOSX 10.6+
        CACHE STRING "Linker flags for security hardening")
    if(NOT DEVELOPER_BUILD)     # Strip debug symbols
        list(APPEND HARDENING_CFLAGS -O2)
        list(APPEND HARDENING_LDFLAGS -S)
    endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_definitions(_FORTIFY_SOURCE=2)     # Buffer overflow detection
    set(HARDENING_CFLAGS
        -fstack-protector-strong    # Stack-based buffer overrun detection
        -Wformat -Wformat-security  # Format string vulnerability
        CACHE STRING "Compiler flags for security hardening")
    set(HARDENING_LDFLAGS
        -Wl,-z,noexecstack   # Stack execution protection
        -Wl,-z,relro,-z,now  # Data relocation protection
        -pie                 # Position independent executable
        CACHE STRING "Linker flags for security hardening")
    if(NOT DEVELOPER_BUILD)     # Strip debug symbols
        list(APPEND HARDENING_CFLAGS -O2)
        list(APPEND HARDENING_LDFLAGS -strip-debug)
    endif()
else()
    message(WARNING "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}. No security "
    "flags set")
endif()
include(CheckCXXCompilerFlag)
foreach(FLAG ${HARDENING_CFLAGS})
    string(MAKE_C_IDENTIFIER "${FLAG}" FLAGRESULT)
    check_cxx_compiler_flag("${FLAG}" FLAG${FLAGRESULT})
    if (NOT FLAG${FLAGRESULT})
        list(REMOVE_ITEM HARDENING_CFLAGS ${FLAG})
        message(WARNING "Compiler does not support security option ${FLAG}")
    endif()
endforeach()
include(CheckLinkerFlag)
foreach(FLAG ${HARDENING_LDFLAGS})
    string(MAKE_C_IDENTIFIER "${FLAG}" FLAGRESULT)
    check_linker_flag(CXX "${FLAG}" FLAG${FLAGRESULT})              # cmake 3.18+
    if (NOT FLAG${FLAGRESULT})
        list(REMOVE_ITEM HARDENING_LDFLAGS ${FLAG})
        message(WARNING "Linker does not support security option ${FLAG}")
    endif()
endforeach()
message(STATUS "Using security hardening compiler flags: ${HARDENING_CFLAGS} and
linker flags: ${HARDENING_LDFLAGS}")
list(TRANSFORM HARDENING_LDFLAGS REPLACE "-pie"
    "$<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>")
string(REPLACE ";" "," CUDA_HARDENING_CFLAGS "${HARDENING_CFLAGS}")
string(REPLACE ";" "," CUDA_HARDENING_LDFLAGS "${HARDENING_LDFLAGS}")
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${CUDA_HARDENING_CFLAGS}>"
    "$<$<COMPILE_LANGUAGE:CXX>:${HARDENING_CFLAGS}>"
    )
add_link_options(
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${CUDA_HARDENING_LDFLAGS}>"
    "$<$<COMPILE_LANGUAGE:CXX>:${HARDENING_LDFLAGS}>"
    )

# Build CUDA module by default if CUDA is available
if(BUILD_CUDA_MODULE)
    find_package(CUDAToolkit REQUIRED)

    if(BUILD_COMMON_CUDA_ARCHS)
        # Build with all known (and deprecated) architectures.
        # See https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#building-for-maximum-compatibility.
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.1")
            set(CMAKE_CUDA_ARCHITECTURES 35-real 37-real 50-real 52-real 53-real
                60-real 61-real 62-real 70-real 72-real 75-real 80-real 86)
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
            set(CMAKE_CUDA_ARCHITECTURES 35-real 37-real 50-real 52-real 53-real
                60-real 61-real 62-real 70-real 72-real 75-real 80)
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "10.0")
            set(CMAKE_CUDA_ARCHITECTURES 30-real 32-real 35-real 37-real 50-real
                52-real 53-real 60-real 61-real 62-real 70-real 72-real 75)
        else()
            message(STATUS "Old CUDA <10.0 detected. Falling back to CMake's default architectures.")
        endif()
    else()
        if(CMAKE_CUDA_ARCHITECTURES)
            message(STATUS "Building with user-provided architectures")
        else()
            file(WRITE
                "${CMAKE_CURRENT_BINARY_DIR}/cuda_architectures.c"
                "
                #include <stdio.h>
                #include <cuda_runtime_api.h>
                int main() {
                    int n;
                    if (cudaGetDeviceCount(&n) == cudaSuccess) {
                        for (int i = 0; i < n; ++i) {
                            int major, minor;
                            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                                    i) == cudaSuccess &&
                                cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                                    i) == cudaSuccess) {
                                if (i > 0) {
                                    printf(\";\");
                                }
                                printf(\"%d%d-real\", major, minor);
                            }
                        }
                    }
                    return 0;
                }
                ")
            try_run(
                DETECTION_RETURN_VALUE DETECTION_COMPILED
                "${CMAKE_CURRENT_BINARY_DIR}"
                "${CMAKE_CURRENT_BINARY_DIR}/cuda_architectures.c"
                LINK_LIBRARIES CUDA::cudart
                RUN_OUTPUT_VARIABLE DETECTED_ARCHITECTURES)
            if(DETECTED_ARCHITECTURES)
                message(STATUS "Building with detected architectures.")
                set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_ARCHITECTURES})
            else()
                message(STATUS "Failed to detect architectures. Falling back to CMake's default architectures.")
            endif()
        endif()
    endif()

    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    enable_language(CUDA)
endif ()

# OS specific settings
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
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj;/Ob2>")
    endif()
    if (STATIC_WINDOWS_RUNTIME)
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    else()
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
endif()

# Downloads files from https://github.com/intel-isl/open3d_downloads to
# Open3D/examples/test_data/open3d_downloads.
include(ExternalProject)
ExternalProject_Add(
    open3d_downloads
    DOWNLOAD_COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/examples/test_data/download_test_data.py
    UPDATE_COMMAND   ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/examples/test_data/download_test_data.py
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

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
    target_compile_features(${target} PUBLIC cxx_std_14)
    # Colorize GCC/Clang terminal outputs
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>)
    endif ()
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp>
        $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}>
    )
    if(BUILD_CUDA_MODULE)
        target_compile_definitions(${target} PRIVATE BUILD_CUDA_MODULE)
        if(BUILD_CACHED_CUDA_MANAGER)
            target_compile_definitions(${target} PRIVATE BUILD_CACHED_CUDA_MANAGER)
        endif()
    endif()
    if(BUILD_GUI)
        target_compile_definitions(${target} PRIVATE BUILD_GUI)
    endif()
    if(USE_BLAS)
        target_compile_definitions(${target} PRIVATE USE_BLAS)
    endif()
    if(BUILD_RPC_INTERFACE)
        target_compile_definitions(${target} PRIVATE BUILD_RPC_INTERFACE ZMQ_STATIC)
    endif()
    if(GLIBCXX_USE_CXX11_ABI)
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
    else()
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
    endif()
    if(ENABLE_HEADLESS_RENDERING)
        target_compile_definitions(${target} PRIVATE HEADLESS_RENDERING)
    endif()
    if(NOT WITH_OPENMP)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>")
    endif()
    if(BUILD_WEBRTC)
        target_compile_definitions(${target} PRIVATE BUILD_WEBRTC)
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
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/EHsc /wd4996>)
            # Multi-thread compile, two ways to enable
            # Option 1, at build time: cmake --build . --parallel %NUMBER_OF_PROCESSORS%
            # https://stackoverflow.com/questions/36633074/set-the-number-of-threads-in-a-cmake-build
            # Option 2, at configure time: add /MP flag, no need to use Option 1
            # https://docs.microsoft.com/en-us/cpp/build/reference/mp-build-with-multiple-processes?view=vs-2019
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP>)
            if(BUILD_GUI)
                # GLEW and Open3D make direct OpenGL calls and link to opengl32.lib;
                # Filament needs to link through bluegl.lib.
                # See https://github.com/google/filament/issues/652
                target_link_options(${target} PRIVATE /force:multiple)
            endif()
            # The examples' .pdb files use up a lot of space and cause us to run
            # out of space on Github Actions. Compressing gives us another couple of GB.
            target_link_options(${target} PRIVATE /pdbcompress)
        endif()
    elseif(APPLE)
        target_compile_definitions(${target} PRIVATE UNIX APPLE)
        target_compile_options(${target} PRIVATE "-Wno-deprecated-declarations")
    elseif(UNIX)
        target_compile_definitions(${target} PRIVATE UNIX)
        target_compile_options(${target} PRIVATE "-Wno-deprecated-declarations" "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-result>")
    endif()
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>")
    # TBB static version is used
    # See: https://github.com/wjakob/tbb/commit/615d690c165d68088c32b6756c430261b309b79c
    target_compile_definitions(${target} PRIVATE __TBB_LIB_NAME=tbb_static)
    if(WITH_FAISS)
        target_compile_definitions(${target} PRIVATE WITH_FAISS)
    endif()
    # Download test data files from open3d_downloads repo.
    add_dependencies(${target} open3d_downloads)
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
    ${CMAKE_CURRENT_SOURCE_DIR}/util/check_style.py
    COMMENT "Python executable used for style check: ${PYTHON_EXECUTABLE}."
)

# `make apply-style` applies style for c++/cuda/python/ipynb files
add_custom_target(apply-style
    COMMAND ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/util/check_style.py --do_apply_style
    COMMENT "Python executable used for style check: ${PYTHON_EXECUTABLE}."
)

# `make check-cpp-style` checks style for c++/cuda files.
# This works outside of python virtualenv.
add_custom_target(check-cpp-style
    COMMAND ${CMAKE_COMMAND}
    -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    -DDO_APPLY_STYLE=OFF
    -P ${CMAKE_CURRENT_SOURCE_DIR}/util/check_cpp_style.cmake
)

# `make apply-cpp-style` applies style for c++/cuda files.
# This works outside of python virtualenv.
add_custom_target(apply-cpp-style
    COMMAND ${CMAKE_COMMAND}
    -DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    -DDO_APPLY_STYLE=ON
    -P ${CMAKE_CURRENT_SOURCE_DIR}/util/check_cpp_style.cmake
)

function(open3d_aligned_print first second)
    string(LENGTH "${first}" first_len)
    math(EXPR star_len "40 - ${first_len}")
    # CMake 3.15+ has string(REPEAT), but this is backwards compatible
    string(SUBSTRING "........................................" 0 ${star_len} stars)
    message(STATUS "  ${first} ${stars} ${second}")
endfunction()

message(STATUS "================================================================================")
message(STATUS "Open3D ${PROJECT_VERSION} Configuration Summary")
message(STATUS "================================================================================")
message(STATUS "Enabled Features:")
open3d_aligned_print("OpenMP" "${WITH_OPENMP}")
open3d_aligned_print("Headless Rendering" "${ENABLE_HEADLESS_RENDERING}")
open3d_aligned_print("Azure Kinect Support" "${BUILD_AZURE_KINECT}")
open3d_aligned_print("Intel RealSense Support" "${BUILD_LIBREALSENSE}")
open3d_aligned_print("CUDA Support" "${BUILD_CUDA_MODULE}")
open3d_aligned_print("Build GUI" "${BUILD_GUI}")
open3d_aligned_print("Build WebRTC visualizer" "${BUILD_WEBRTC}")
open3d_aligned_print("Build Shared Library" "${BUILD_SHARED_LIBS}")
if(WIN32)
   open3d_aligned_print("Use Windows Static Runtime" "${STATIC_WINDOWS_RUNTIME}")
endif()
open3d_aligned_print("Build Unit Tests" "${BUILD_UNIT_TESTS}")
open3d_aligned_print("Build Examples" "${BUILD_EXAMPLES}")
open3d_aligned_print("Build Python Module" "${BUILD_PYTHON_MODULE}")
open3d_aligned_print("Build Jupyter Extension" "${BUILD_JUPYTER_EXTENSION}")
open3d_aligned_print("Build TensorFlow Ops" "${BUILD_TENSORFLOW_OPS}")
open3d_aligned_print("Build PyTorch Ops" "${BUILD_PYTORCH_OPS}")
if (BUILD_PYTORCH_OPS AND BUILD_CUDA_MODULE AND
    CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0" )
    message( WARNING
        "--------------------------------------------------------------------------------\n"
        "                                                                                \n"
        " You are compiling PyTorch ops with CUDA 11. This configuration may have        \n"
        " stability issues. See https://github.com/intel-isl/Open3D/issues/3324 and      \n"
        " https://github.com/pytorch/pytorch/issues/52663 for more information on this   \n"
        " problem.                                                                       \n"
        "                                                                                \n"
        " We recommend to compile PyTorch from source with compile flags                 \n"
        "   '-Xcompiler -fno-gnu-unique'                                                 \n"
        "                                                                                \n"
        " or use the PyTorch wheels at                                                   \n"
        "   https://github.com/intel-isl/open3d_downloads/releases/tag/torch1.7.1        \n"
        "                                                                                \n"
        "--------------------------------------------------------------------------------\n"
    )
endif()
open3d_aligned_print("Build Benchmarks" "${BUILD_BENCHMARKS}")
open3d_aligned_print("Bundle Open3D-ML" "${BUNDLE_OPEN3D_ML}")
open3d_aligned_print("Build RPC interface" "${BUILD_RPC_INTERFACE}")
if(GLIBCXX_USE_CXX11_ABI)
    set(usage "1")
else()
    set(usage "0")
endif()
open3d_aligned_print("Force GLIBCXX_USE_CXX11_ABI=" "${usage}")
message(STATUS "================================================================================")
message(STATUS "Third-Party Dependencies:")
set(deps
    EIGEN3
    FAISS
    FILAMENT
    FLANN
    FMT
    GLEW
    GLFW
    GOOGLETEST
    IMGUI
    IPPICV
    JPEG
    JSONCPP
    LIBLZF
    OPENGL
    PNG
    PYBIND11
    QHULL
    LIBREALSENSE
    TINYFILEDIALOGS
    TINYGLTF
    TINYOBJLOADER
    WEBRTC
)
foreach(dep IN ITEMS ${deps})
    if(${dep}_TARGET)
        if(NOT USE_SYSTEM_${dep})
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
message(STATUS "================================================================================")
