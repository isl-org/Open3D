#
# Open3D 3rd party library integration
#
set(Open3D_3RDPARTY_DIR "${PROJECT_SOURCE_DIR}/3rdparty")

# EXTERNAL_MODULES
# CMake modules we depend on in our public interface. These are modules we
# need to find_package() in our CMake config script, because we will use their
# targets.
set(Open3D_3RDPARTY_EXTERNAL_MODULES)

# PUBLIC_TARGETS
# CMake targets we link against in our public interface. They are
# either locally defined and installed, or imported from an external module
# (see above).
set(Open3D_3RDPARTY_PUBLIC_TARGETS)

# HEADER_TARGETS
# CMake targets we use in our public interface, but as a special case we do not
# need to link against the library. This simplifies dependencies where we merely
# expose declared data types from other libraries in our public headers, so it
# would be overkill to require all library users to link against that dependency.
set(Open3D_3RDPARTY_HEADER_TARGETS)

# PRIVATE_TARGETS
# CMake targets for dependencies which are not exposed in the public API. This
# will probably include HEADER_TARGETS, but also anything else we use internally.
set(Open3D_3RDPARTY_PRIVATE_TARGETS)

# Unforunately, CMake <3.17 does not propagates link directories for transitive
# linkage of static libraries. This is set to TRUE if we need the link directory
# on the Open3D main library as a workaround.
set(Open3D_NEED_LINK_DIRECTORY FALSE)

find_package(PkgConfig QUIET)

function(build_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER" "DIRECTORY" "INCLUDE_DIRS;SOURCES;LIBS" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Invalid syntax: build_3rdparty_library(${name} ${ARGN})")
    endif()
    if(NOT arg_DIRECTORY)
        set(arg_DIRECTORY "${name}")
    endif()
    if(arg_INCLUDE_DIRS)
        set(include_dirs)
        foreach(incl IN LISTS arg_INCLUDE_DIRS)
            list(APPEND include_dirs "${Open3D_3RDPARTY_DIR}/${arg_DIRECTORY}/${incl}")
        endforeach()
    else()
        set(include_dirs "${Open3D_3RDPARTY_DIR}/${arg_DIRECTORY}/")
    endif()
    message(STATUS "Building third-party library ${name} from source")
    if(arg_SOURCES)
        set(sources)
        foreach(src ${arg_SOURCES})
            list(APPEND sources "${Open3D_3RDPARTY_DIR}/${arg_DIRECTORY}/${src}")
        endforeach()
        add_library(${name} STATIC ${sources})
        foreach(incl IN LISTS include_dirs)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif()
            target_include_directories(${name} SYSTEM PUBLIC 
                $<BUILD_INTERFACE:${incl_path}>
            )
        endforeach()
        target_include_directories(${name} PUBLIC 
            $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}/3rdparty>
        )
        set_target_properties(${name} PROPERTIES
            OUTPUT_NAME "${PROJECT_NAME}_${name}"
            POSITION_INDEPENDENT_CODE ON
        )
        if(arg_LIBS)
            target_link_libraries(${name} PRIVATE ${arg_LIBS})
        endif()
    else()
        add_library(${name} INTERFACE)
        foreach(incl IN LISTS include_dirs)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif()
            target_include_directories(${name} SYSTEM INTERFACE 
                $<BUILD_INTERFACE:${incl_path}>
            )
        endforeach()
        target_include_directories(${name} INTERFACE
            $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}/3rdparty>
        )
    endif()
    if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets
            RUNTIME DESTINATION ${Open3D_INSTALL_BIN_DIR}
            ARCHIVE DESTINATION ${Open3D_INSTALL_LIB_DIR}
            LIBRARY DESTINATION ${Open3D_INSTALL_LIB_DIR}
        )
    endif()
    if(arg_PUBLIC OR arg_HEADER)
        foreach(incl IN LISTS include_dirs)
            install(DIRECTORY ${incl}
                DESTINATION ${Open3D_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}/3rdparty
                FILES_MATCHING
                    PATTERN "*.h"
                    PATTERN "*.hpp"
            )
        endforeach()
    endif()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()

function(pkg_config_3rdparty_library name)
    string(TOUPPER "${name}" name_uc)
    if(PKGCONFIG_FOUND)
        pkg_search_module(${name_uc} ${ARGN})
    endif()
    if(${name_uc}_FOUND)
        message(STATUS "Using installed third-party library ${name} ${${name_uc}_VERSION}")
        add_library(${name} INTERFACE)
        target_include_directories(${name} SYSTEM INTERFACE ${${name_uc}_INCLUDE_DIRS})
        foreach(lib IN LISTS ${name_uc}_LIBRARIES)
            find_library(${lib}_LIBRARY NAMES ${lib} PATHS ${${name_uc}_LIBRARY_DIRS})
            target_link_libraries(${name} INTERFACE ${${lib}_LIBRARY})
        endforeach()
        foreach(flag IN LISTS ${name_uc}_CFLAGS ${name_uc}_CFLAGS_OTHER)
            if(flag MATCHES "-D.*")
                target_compile_definitions(${name} INTERFACE ${flag})
            endif()
        endforeach()
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
        set(${name_uc}_FOUND TRUE PARENT_SCOPE)
        add_library(${PROJECT_NAME}::${name} ALIAS ${name})
    else()
        message(STATUS "Unable to find installed third-party library ${name}")
        set(${name_uc}_FOUND FALSE PARENT_SCOPE)
    endif()
endfunction()

function(import_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER" "DIRECTORY;INCLUDE_DIR;LIBRARY" "" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Invalid syntax: import_3rdparty_library(${name} ${ARGN})")
    endif()
    if(NOT arg_DIRECTORY)
        set(arg_DIRECTORY "${name}")
    endif()
    add_library(${name} INTERFACE)
    if(arg_INCLUDE_DIR)
        if (arg_INCLUDE_DIR MATCHES "(.*)/$")
            set(incl_path ${CMAKE_MATCH_1})
        else()
            get_filename_component(incl_path "${incl}" DIRECTORY)
        endif()
        target_include_directories(${name} INTERFACE $<BUILD_INTERFACE:${incl_path}>)
        if(arg_PUBLIC OR arg_HEADER)
            install(DIRECTORY ${arg_INCLUDE_DIR} DESTINATION ${Open3D_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}/3rdparty
                FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
            )
            target_include_directories(${name} INTERFACE $<INSTALL_INTERFACE:${Open3D_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}/3rdparty>)
        endif()
    endif()
    if(arg_LIBRARY)
        set(library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${arg_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX})
        set(installed_library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}_${name}${CMAKE_STATIC_LIBRARY_SUFFIX})
        set_target_properties(${name} PROPERTIES INTERFACE_LINK_DIRECTORIES $<BUILD_INTERFACE:${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}>)
        target_link_libraries(${name} INTERFACE $<BUILD_INTERFACE:-l${arg_LIBRARY}>)
        if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
            install(FILES ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/${library_filename}
                DESTINATION ${Open3D_INSTALL_LIB_DIR}
                RENAME ${installed_library_filename}
            )
            target_link_libraries(${name} INTERFACE $<INSTALL_INTERFACE:${PROJECT_NAME}_${name}>)
            set(Open3D_NEED_LINK_DIRECTORY TRUE PARENT_SCOPE)
        endif()
    endif()
    if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
    endif()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()

# OpenMP
if(WITH_OPENMP)
    find_package(OpenMP)
    if(TARGET OpenMP::OpenMP_CXX)
        message(STATUS "Building with OpenMP")
        list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "OpenMP::OpenMP_CXX")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "OpenMP")
        endif()
    endif()
endif()

# Dirent
if(WIN32)
    message(STATUS "Building DIRENT from source (WIN32)")
    build_3rdparty_library(ext_dirent DIRECTORY dirent)
    set(DIRENT_TARGET "ext_dirent")
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${DIRENT_TARGET}")
endif()

# Eigen3
if(NOT BUILD_EIGEN3)
    find_package(Eigen3)
    if(TARGET Eigen3::Eigen)
        message(STATUS "Using installed third-party library Eigen3 ${EIGEN3_VERSION_STRING}")
        # Eigen3 is a publicly visible dependency, so add it to the list of
        # modules we need to find in the Open3D config script.
        list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "Eigen3")
        set(EIGEN3_TARGET "Eigen3::Eigen")
    else()
        message(STATUS "Unable to find installed third-party library Eigen3")
        set(BUILD_EIGEN3 ON)
    endif()
endif()
if(BUILD_EIGEN3)
    build_3rdparty_library(ext_eigen3 PUBLIC DIRECTORY Eigen INCLUDE_DIRS Eigen)
    set(EIGEN3_TARGET "ext_eigen3")
endif()
list(APPEND Open3D_3RDPARTY_PUBLIC_TARGETS "${EIGEN3_TARGET}")

# Flann
if(NOT BUILD_FLANN)
    pkg_config_3rdparty_library(ext_flann flann)
endif()
if(BUILD_FLANN OR NOT FLANN_FOUND)
    build_3rdparty_library(ext_flann DIRECTORY flann)
endif()
set(FLANN_TARGET "ext_flann")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${FLANN_TARGET}")

# GLEW
if(NOT BUILD_GLEW)
    find_package(GLEW)
    if(TARGET GLEW::GLEW)
        message(STATUS "Using installed third-party library GLEW ${GLEW_VERSION}")
        list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "GLEW")
        set(GLEW_TARGET "GLEW::GLEW")
    else()
        pkg_config_3rdparty_library(ext_glew glew)
        if(GLEW_FOUND)
            set(GLEW_TARGET "ext_glew")
        else()
            set(BUILD_GLEW ON)
        endif()
    endif()
endif()
if(BUILD_GLEW)
    build_3rdparty_library(ext_glew HEADER DIRECTORY glew SOURCES src/glew.c INCLUDE_DIRS include/)
    set(GLEW_TARGET "ext_glew")
endif()
list(APPEND Open3D_3RDPARTY_HEADER_TARGETS "${GLEW_TARGET}")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${GLEW_TARGET}")

# GLFW
if(NOT BUILD_GLFW)
    find_package(glfw3)
    if(TARGET glfw)
        message(STATUS "Using installed third-party library glfw3")
        list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "glfw3")
        set(GLFW_TARGET "glfw")
    else()
        pkg_config_3rdparty_library(ext_glfw3 glfw3)
        if(GLFW_FOUND)
            set(GLFW_TARGET "ext_glfw3")
        else()
            set(BUILD_GLFW ON)
        endif()
    endif()
endif()
if(BUILD_GLFW)
    message(STATUS "Building third-party library ext_glfw3 from source")
    add_subdirectory(${Open3D_3RDPARTY_DIR}/GLFW)
    if (BUILD_SHARED_LIBS AND UNIX)
        set(GLFW_LIB_NAME glfw)
    else()
        set(GLFW_LIB_NAME glfw3)
    endif()
    import_3rdparty_library(ext_glfw3 HEADER INCLUDE_DIR ${Open3D_3RDPARTY_DIR}/GLFW/include/ LIBRARY ${GLFW_LIB_NAME})
    add_dependencies(ext_glfw3 glfw)
    if(UNIX AND NOT APPLE)
        find_package(X11 QUIET)
        if(X11_FOUND)
            target_link_libraries(ext_glfw3 INTERFACE ${X11_X11_LIB} ${CMAKE_THREAD_LIBS_INIT})
        endif()
        find_library(RT_LIBRARY rt)
        if(RT_LIBRARY)
            target_link_libraries(ext_glfw3 INTERFACE ${RT_LIBRARY})
        endif()
        find_library(MATH_LIBRARY m)
        if(MATH_LIBRARY)
            target_link_libraries(ext_glfw3 INTERFACE ${MATH_LIBRARY})
        endif()
        if(CMAKE_DL_LIBS)
            target_link_libraries(ext_glfw3 INTERFACE ${CMAKE_DL_LIBS})
        endif()
    endif()
    set(GLFW_TARGET "ext_glfw3")
endif()
list(APPEND Open3D_3RDPARTY_HEADER_TARGETS "${GLFW_TARGET}")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${GLFW_TARGET}")

# Headless rendering
if (ENABLE_HEADLESS_RENDERING)
    find_package(OSMesa REQUIRED)
    add_library(ext_osmesa INTERFACE)
    target_include_directories(ext_osmesa INTERFACE ${OSMESA_INCLUDE_DIR})
    target_link_libraries(ext_osmesa INTERFACE ${OSMESA_LIBRARY})
    if(NOT BUILD_SHARED_LIBS)
        install(TARGETS ext_osmesa EXPORT ${PROJECT_NAME}Targets
        RUNTIME DESTINATION ${Open3D_INSTALL_BIN_DIR}
        ARCHIVE DESTINATION ${Open3D_INSTALL_LIB_DIR}
        LIBRARY DESTINATION ${Open3D_INSTALL_LIB_DIR}
    )
    endif()
    set(OSMESA_TARGET "ext_osmesa")
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${OSMESA_TARGET}")
endif()

find_package(OpenGL)
if(TARGET OpenGL::GL)
    if(NOT BUILD_SHARED_LIBS)
        list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "OpenGL")
    endif()
    set(OPENGL_TARGET "OpenGL::GL")
endif()
if(OPENGL_TARGET)
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${OPENGL_TARGET}")
endif()

# Pybind11
if(NOT BUILD_PYBIND11)
    find_package(pybind11)
endif()
if (BUILD_PYBIND11 OR NOT TARGET pybind11::module)
    add_subdirectory(${Open3D_3RDPARTY_DIR}/pybind11)
endif()

# Azure Kinect
include(${Open3D_3RDPARTY_DIR}/azure_kinect/azure_kinect.cmake)

# jsoncpp
if(NOT BUILD_JSONCPP)
    find_package(jsoncpp)
    if(TARGET jsoncpp_lib)
        message(STATUS "Using installed third-party library jsoncpp")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "jsoncpp")
        endif()
        set(JSONCPP_TARGET "jsoncpp_lib")
    elseif(TARGET jsoncpp)
        message(STATUS "Using installed third-party library jsoncpp")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "jsoncpp")
        endif()
        set(JSONCPP_TARGET "jsoncpp")
    else()
        message(STATUS "Unable to find installed third-party library jsoncpp")
        set(BUILD_JSONCPP ON)
    endif()
endif()
if(BUILD_JSONCPP)
    build_3rdparty_library(ext_jsoncpp DIRECTORY jsoncpp
        SOURCES
            json_reader.cpp
            json_value.cpp
            json_writer.cpp
        INCLUDE_DIRS
            include/
    )
    set(JSONCPP_TARGET "ext_jsoncpp")
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${JSONCPP_TARGET}")

# liblzf
if(NOT BUILD_LIBLZF)
    find_package(liblzf)
    if(TARGET liblzf::liblzf)
        message(STATUS "Using installed third-party library liblzf")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "liblzf")
        endif()
        set(LIBLZF_TARGET "liblzf::liblzf")
    else()
        message(STATUS "Unable to find installed third-party library liblzf")
        set(BUILD_LIBLZF ON)
    endif()
endif()
if(BUILD_LIBLZF)
    build_3rdparty_library(ext_lzf DIRECTORY liblzf
        SOURCES
            liblzf/lzf_c.c
            liblzf/lzf_d.c
    )
    set(LIBLZF_TARGET "ext_lzf")
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${LIBLZF_TARGET}")

# tritriintersect
build_3rdparty_library(ext_tritriintersect DIRECTORY tomasakeninemoeller INCLUDE_DIRS include/)
set(TRITRIINTERSECT_TARGET "ext_tritriintersect")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${TRITRIINTERSECT_TARGET}")

# PNG
if(NOT BUILD_PNG)
    find_package(PNG)
    if(TARGET PNG::PNG)
        message(STATUS "Using installed third-party library libpng")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "PNG")
        endif()
        set(PNG_TARGET "PNG::PNG")
    else()
        message(STATUS "Unable to find installed third-party library libpng")
        set(BUILD_PNG ON)
    endif()
endif()
if(BUILD_PNG)
    message(STATUS "Building third-party library libpng from source")
    add_subdirectory(${Open3D_3RDPARTY_DIR}/libpng)
    import_3rdparty_library(ext_png INCLUDE_DIR ${Open3D_3RDPARTY_DIR}/libpng/ LIBRARY ${PNG_LIBRARIES})
    add_dependencies(ext_png ${PNG_LIBRARIES})
    set(PNG_TARGET "ext_png")
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${PNG_TARGET}")

# zlib
if(NOT BUILD_ZLIB)
    find_package(ZLIB)
    if(TARGET ZLIB::ZLIB)
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "ZLIB")
        endif()
        set(ZLIB_TARGET "ZLIB::ZLIB")
    else()
        set(BUILD_ZLIB ON)
    endif()
endif()
if(BUILD_ZLIB)
    add_subdirectory(${Open3D_3RDPARTY_DIR}/zlib)
    import_3rdparty_library(ext_zlib INCLUDE_DIR ${Open3D_3RDPARTY_DIR}/zlib LIBRARY ${ZLIB_LIBRARY})
    add_dependencies(ext_zlib ${ZLIB_LIBRARY})
    set(ZLIB_TARGET "ext_zlib")
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${ZLIB_TARGET}")

if(NOT BUILD_JPEG)
    find_package(JPEG)
    if(TARGET JPEG::JPEG)
        message(STATUS "Using installed third-party library JPEG")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "JPEG")
        endif()
        set(JPEG_TARGET "JPEG::JPEG")
    else()
        message(STATUS "Unable to find installed third-party library JPEG")
        set(BUILD_JPEG ON)
    endif()
endif()
if (BUILD_JPEG)
    message(STATUS "Building third-party library JPEG from source")
    include(${Open3D_3RDPARTY_DIR}/libjpeg-turbo/libjpeg-turbo.cmake)
    import_3rdparty_library(ext_jpeg INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo-install/include/ LIBRARY ${JPEG_TURBO_LIBRARIES})
    add_dependencies(ext_jpeg ext_turbojpeg)
    set(JPEG_TARGET "ext_jpeg")
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${JPEG_TARGET}")

# RealSense
if (BUILD_LIBREALSENSE)
    message(STATUS "Building third-party library librealsense from source")
    add_subdirectory(${Open3D_3RDPARTY_DIR}/librealsense)
    import_3rdparty_library(ext_realsense INCLUDE_DIR ${Open3D_3RDPARTY_DIR}/librealsense/include/ LIBRARY ${REALSENSE_LIBRARY})
    add_dependencies(ext_realsense ${REALSENSE_LIBRARY})
    set(REALSENSE_TARGET "ext_realsense")
    list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${REALSENSE_TARGET}")
endif ()

# tinyfiledialogs
build_3rdparty_library(ext_tinyfiledialogs DIRECTORY tinyfiledialogs SOURCES include/tinyfiledialogs/tinyfiledialogs.c INCLUDE_DIRS include/)
set(TINYFILEDIALOGS_TARGET "ext_tinyfiledialogs")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${TINYFILEDIALOGS_TARGET}")

# tinygltf
if(NOT BUILD_TINYGLTF)
    find_package(TinyGLTF)
    if(TARGET TinyGLTF::TinyGLTF)
        message(STATUS "Using installed third-party library TinyGLTF")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "TinyGLTF")
        endif()
        set(TINYGLTF_TARGET "TinyGLTF::TinyGLTF")
    else()
        message(STATUS "Unable to find installed third-party library TinyGLTF")
        set(BUILD_TINYGLTF ON)
    endif()
endif()
if(BUILD_TINYGLTF)
    build_3rdparty_library(ext_tinygltf DIRECTORY tinygltf INCLUDE_DIRS tinygltf/)
    target_compile_definitions(ext_tinygltf INTERFACE TINYGLTF_IMPLEMENTATION STB_IMAGE_IMPLEMENTATION STB_IMAGE_WRITE_IMPLEMENTATION)
    set(TINYGLTF_TARGET "ext_tinygltf")
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${TINYGLTF_TARGET}")

# tinyobjloader
if(NOT BUILD_TINYOBJLOADER)
    find_package(tinyobjloader)
    if(TARGET tinyobjloader::tinyobjloader)
        message(STATUS "Using installed third-party library tinyobjloader")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "tinyobjloader")
        endif()
        set(TINYOBJLOADER_TARGET "tinyobjloader::tinyobjloader")
    else()
        message(STATUS "Unable to find installed third-party library tinyobjloader")
        set(BUILD_TINYOBJLOADER ON)
    endif()
endif()
if(BUILD_TINYOBJLOADER)
    build_3rdparty_library(ext_tinyobjloader DIRECTORY tinyobjloader INCLUDE_DIRS tinyobjloader/)
    target_compile_definitions(ext_tinyobjloader INTERFACE TINYOBJLOADER_IMPLEMENTATION)
    set(TINYOBJLOADER_TARGET "ext_tinyobjloader")
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${TINYOBJLOADER_TARGET}")

# rply
build_3rdparty_library(ext_rply DIRECTORY rply SOURCES rply/rply.c INCLUDE_DIRS rply/)
set(RPLY_TARGET "ext_rply")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${RPLY_TARGET}")

# Qhull
if(NOT BUILD_QHULL)
    find_package(Qhull)
    if(TARGET Qhull::qhullcpp)
        message(STATUS "Using installed third-party library Qhull")
        if(NOT BUILD_SHARED_LIBS)
            list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "Qhull")
        endif()
        set(QHULL_TARGET "Qhull::qhullcpp")
    else()
        message(STATUS "Unable to find installed third-party library Qhull")
        set(BUILD_QHULL ON)
    endif()
endif()
if (BUILD_QHULL)
    build_3rdparty_library(ext_qhull_r DIRECTORY qhull
        SOURCES
            src/libqhull_r/global_r.c
            src/libqhull_r/stat_r.c
            src/libqhull_r/geom2_r.c
            src/libqhull_r/poly2_r.c
            src/libqhull_r/merge_r.c
            src/libqhull_r/libqhull_r.c
            src/libqhull_r/geom_r.c
            src/libqhull_r/poly_r.c
            src/libqhull_r/qset_r.c
            src/libqhull_r/mem_r.c
            src/libqhull_r/random_r.c
            src/libqhull_r/usermem_r.c
            src/libqhull_r/userprintf_r.c
            src/libqhull_r/io_r.c
            src/libqhull_r/user_r.c
            src/libqhull_r/rboxlib_r.c
            src/libqhull_r/userprintf_rbox_r.c
        INCLUDE_DIRS
            src/
    )
    build_3rdparty_library(ext_qhullcpp DIRECTORY qhull
        SOURCES
            src/libqhullcpp/Coordinates.cpp
            src/libqhullcpp/PointCoordinates.cpp
            src/libqhullcpp/Qhull.cpp
            src/libqhullcpp/QhullFacet.cpp
            src/libqhullcpp/QhullFacetList.cpp
            src/libqhullcpp/QhullFacetSet.cpp
            src/libqhullcpp/QhullHyperplane.cpp
            src/libqhullcpp/QhullPoint.cpp
            src/libqhullcpp/QhullPointSet.cpp
            src/libqhullcpp/QhullPoints.cpp
            src/libqhullcpp/QhullQh.cpp
            src/libqhullcpp/QhullRidge.cpp
            src/libqhullcpp/QhullSet.cpp
            src/libqhullcpp/QhullStat.cpp
            src/libqhullcpp/QhullVertex.cpp
            src/libqhullcpp/QhullVertexSet.cpp
            src/libqhullcpp/RboxPoints.cpp
            src/libqhullcpp/RoadError.cpp
            src/libqhullcpp/RoadLogEvent.cpp
        INCLUDE_DIRS
            src/
    )
    target_link_libraries(ext_qhullcpp PRIVATE ext_qhull_r)
    set(QHULL_TARGET "ext_qhullcpp")
endif()
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${QHULL_TARGET}")

# fmt
if(NOT BUILD_FMT)
    find_package(fmt)
    if(TARGET fmt::fmt)
        message(STATUS "Using installed third-party library fmt")
        list(APPEND Open3D_3RDPARTY_EXTERNAL_MODULES "fmt")
        set(FMT_TARGET "fmt::fmt")
    else()
        message(STATUS "Unable to find installed third-party library fmt")
        set(BUILD_FMT ON)
    endif()
endif()
if(BUILD_FMT)
    # We set the FMT_HEADER_ONLY macro, so no need to actually compile the source
    build_3rdparty_library(ext_fmt PUBLIC DIRECTORY fmt INCLUDE_DIRS include/)
    set(FMT_TARGET "ext_fmt")
endif()
list(APPEND Open3D_3RDPARTY_HEADER_TARGETS "${FMT_TARGET}")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${FMT_TARGET}")

# Googletest
if (BUILD_UNIT_TESTS)
    message(STATUS "Building googletest from source")
    build_3rdparty_library(ext_googletest DIRECTORY googletest
        SOURCES
            googletest/src/gtest-all.cc
            googlemock/src/gmock-all.cc
        INCLUDE_DIRS
            googletest/include/
            googletest/
            googlemock/include/
            googlemock/
    )
    set(GOOGLETEST_TARGET "ext_googletest")
endif()

# PoissonRecon
build_3rdparty_library(ext_poisson DIRECTORY PoissonRecon INCLUDE_DIRS PoissonRecon)
set(POISSON_TARGET "ext_poisson")
list(APPEND Open3D_3RDPARTY_PRIVATE_TARGETS "${POISSON_TARGET}")

# benchmark
if(BUILD_BENCHMARKS)
    # turn off installing and testing of the benchmark lib
    set(BENCHMARK_ENABLE_INSTALL  OFF CACHE BOOL "This should be OFF. Enables installing the benchmark lib")
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "This should be OFF. Enables gtest framework for the benchmark lib")
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "This should be OFF. Enables tests for the benchmark lib")
    add_subdirectory(${Open3D_3RDPARTY_DIR}/benchmark)
    # set the cache vars introduced by the benchmark lib as advanced to not
    # clutter the cmake interfaces
    mark_as_advanced(
        BENCHMARK_ENABLE_INSTALL 
        BENCHMARK_ENABLE_GTEST_TESTS 
        BENCHMARK_ENABLE_TESTING 
        BENCHMARK_ENABLE_ASSEMBLY_TESTS
        BENCHMARK_DOWNLOAD_DEPENDENCIES
        BENCHMARK_BUILD_32_BITS
        BENCHMARK_ENABLE_EXCEPTIONS
        BENCHMARK_ENABLE_LTO
        BENCHMARK_USE_LIBCXX
    )
endif()
