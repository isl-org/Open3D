cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)

project(TestVisualizer)

# Find installed Open3D, this imports
# Targets:
#   Open3D::Open3D
# Variables:
#   Open3D_INCLUDE_DIRS
#   Open3D_LIBRARIES
#   Open3D_LIBRARY_DIRS
#   Open3D_BINARY_DIR
#   Open3D_CXX_FLAGS
if(WIN32)
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/CMake)
else()
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
endif()

# Set OS-specific things here
if(WIN32)
elseif(CYGWIN)
elseif(APPLE)
elseif(UNIX)
	add_definitions(-DUNIX)
	add_compile_options(-Wno-deprecated-declarations)
	add_compile_options(-Wno-unused-result)
    add_definitions(-O3)
endif(WIN32)

# Open3D
if (TARGET Open3D::Open3D)
    message(STATUS "Found Open3D ${Open3D_VERSION}")

    add_executable(TestVisualizer TestVisualizer.cpp)
    target_link_libraries(TestVisualizer Open3D::Open3D)
    if(NOT WIN32)
        target_link_libraries(TestVisualizer ${CMAKE_DL_LIBS})
    endif()

    # Windows: copy .dll next to the executable
    if(WIN32)
        get_target_property(Open3D_Lib_Type Open3D::Open3D TYPE)
        if(${Open3D_Lib_Type} STREQUAL "SHARED_LIBRARY")
            message("Will copy Open3D.dll next to .exe")
            add_custom_command(TARGET TestVisualizer POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                    $<TARGET_FILE:Open3D::Open3D>
                                    $<TARGET_FILE_DIR:TestVisualizer>)
        endif()
    endif()

else ()
    message(SEND_ERROR "Open3D not found")
endif ()
