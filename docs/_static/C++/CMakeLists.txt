cmake_minimum_required(VERSION 3.0)

project(TestVisualizer)

find_package(PkgConfig QUIET)
find_package(OpenGL    QUIET)
find_package(pybind11  QUIET)
find_package(OpenMP    QUIET)
find_package(Open3D    QUIET HINTS "~/open3d_install/lib/CMake")

if (PKGCONFIG_FOUND)
    pkg_search_module(EIGEN3    eigen3>=3.2.7   QUIET)
    pkg_search_module(JPEG      libjpeg         QUIET)
    pkg_search_module(GLFW      glfw3           QUIET)
    pkg_search_module(GLEW      glew            QUIET)
    pkg_search_module(JSONCPP   jsoncpp>=1.7.0  QUIET)
    pkg_search_module(PNG       libpng>=1.6.0   QUIET)
endif (PKGCONFIG_FOUND)

option(Open3D_USE_NATIVE_DEPENDENCY_BUILD "Prioritize Found dependency build" ON)

# Set OS-specific things here
if(WIN32)
elseif(CYGWIN)
elseif(APPLE)
elseif(UNIX)
	add_definitions(-DUNIX)
	# enable c++11
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -L${Open3D_LIBRARY_DIRS}")
	add_compile_options(-Wno-deprecated-declarations)
	add_compile_options(-Wno-unused-result)
    add_definitions(-O3)
endif(WIN32)

# Open3D
if (Open3D_FOUND)
    message(STATUS "Found Open3D ${Open3D_VERSION}")

    add_executable(TestVisualizer TestVisualizer.cpp)

    include_directories(${Open3D_INCLUDE_DIRS})

    link_directories(${Open3D_LIBRARY_DIRS})

    set_target_properties(TestVisualizer PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

    target_link_libraries(TestVisualizer ${Open3D_LIBRARIES})
    target_link_libraries(TestVisualizer tinyfiledialogs)

else ()
    message(SEND_ERROR "Open3D not found")
endif ()

# Eigen3
if (EIGEN3_FOUND)
    message(STATUS "Found eigen3 ${EIGEN3_VERSION}")
    include_directories(${EIGEN3_INCLUDE_DIRS})
else ()
    message(SEND_ERROR "eigen3 not found")
endif ()

# GLEW
if (GLEW_FOUND)
    message(STATUS "Found glew ${GLEW_VERSION}")
    include_directories(${GLEW_INCLUDE_DIRS})
    target_link_libraries(TestVisualizer ${GLEW_LIBRARIES})
    link_directories(${GLFW_LIBRARY_DIRS})
else ()
    message(SEND_ERROR "glew not found")
endif ()

# GLFW
if (GLFW_FOUND AND OPENGL_FOUND)
    include_directories(${GLFW_INCLUDE_DIRS})
    target_link_libraries(TestVisualizer ${GLFW_LIBRARIES})
    message(STATUS "Found glfw ${GLFW_VERSION}")
else ()
    message(SEND_ERROR "glfw not found")
endif ()

# JPEG
if (JPEG_FOUND)
    message(STATUS "Found libjpeg ${JPEG_VERSION}")
    include_directories(${JPEG_INCLUDE_DIR})
    target_link_libraries(TestVisualizer ${JPEG_LIBRARIES})
else ()
    message(SEND_ERROR "libjpeg not found")
endif ()

# JSON
if (JSONCPP_FOUND)
    message(STATUS "Found jsoncpp ${JSONCPP_VERSION}")
    include_directories(${JSONCPP_INCLUDE_DIRS})
    target_link_libraries(TestVisualizer ${JSONCPP_LIBRARIES})
    link_directories(${JSONCPP_LIBRARY_DIRS})
else ()
    message(SEND_ERROR "jsoncpp not found")
endif ()

# OpenGL
if (OPENGL_FOUND)
    include_directories(${OPENGL_INCLUDE_DIR})
    target_link_libraries(TestVisualizer ${OPENGL_gl_LIBRARY})
    message(STATUS "Found OpenGL")
else ()
    message(SEND_ERROR "OpenGL not found")
endif ()

# OpenMP
if (OPENMP_FOUND)
    message(STATUS "Found OpenMP ${OpenMP_VERSION}")
    target_link_libraries(TestVisualizer ${OMP_LIBRARIES})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(WARNING "OpenMP not found")
endif ()

# PNG
if (PNG_FOUND)
    message(STATUS "Found libpng ${PNG_VERSION}")
    include_directories(${PNG_INCLUDE_DIRS})
    target_link_libraries(TestVisualizer ${PNG_LIBRARIES})
else ()
    message(SEND_ERROR "libpng not found")
endif ()
