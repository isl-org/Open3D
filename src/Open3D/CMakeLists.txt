# Configure a header file to pass the version settings to the source code
configure_file("${PROJECT_SOURCE_DIR}/src/Open3D/Open3D.h.in"
               "${PROJECT_SOURCE_DIR}/src/Open3D/Open3D.h")
configure_file("${PROJECT_SOURCE_DIR}/src/Open3D/Open3DConfig.h.in"
               "${PROJECT_SOURCE_DIR}/src/Open3D/Open3DConfig.h")

# Subdirectories
add_subdirectory(Camera)
add_subdirectory(ColorMap)
add_subdirectory(Geometry)
add_subdirectory(Integration)
add_subdirectory(Odometry)
add_subdirectory(Registration)
add_subdirectory(Utility)
add_subdirectory(IO)
add_subdirectory(Visualization)

# Source group for Visual Studio
ADD_SOURCE_GROUP(Camera)
ADD_SOURCE_GROUP(ColorMap)
ADD_SOURCE_GROUP(Geometry)
ADD_SOURCE_GROUP(Integration)
ADD_SOURCE_GROUP(Odometry)
ADD_SOURCE_GROUP(Registration)
ADD_SOURCE_GROUP(Utility)
ADD_SOURCE_GROUP(IO)
ADD_SOURCE_GROUP(Visualization)

# Install headers
install(DIRECTORY   "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include"
        PATTERN     "Visualization/Shader/GLSL" EXCLUDE
        PATTERN     "*.cpp" EXCLUDE
        PATTERN     "*.in" EXCLUDE
        PATTERN     "*.txt" EXCLUDE)

# note: adding at least one real source file to any target that references
# reference: https://cmake.org/cmake/help/v3.12/command/add_library.html#object-libraries
add_library(${CMAKE_PROJECT_NAME}
    Open3DConfig.h
    Open3DConfig.cpp
    $<TARGET_OBJECTS:Camera>
    $<TARGET_OBJECTS:ColorMap>
    $<TARGET_OBJECTS:Geometry>
    $<TARGET_OBJECTS:Integration>
    $<TARGET_OBJECTS:Odometry>
    $<TARGET_OBJECTS:Registration>
    $<TARGET_OBJECTS:Utility>
    $<TARGET_OBJECTS:IO>
    $<TARGET_OBJECTS:Visualization>
)
ShowAndAbortOnWarning(${CMAKE_PROJECT_NAME})

# headless rendering
if (ENABLE_HEADLESS_RENDERING)
    find_package(OSMesa REQUIRED)
    include_directories(${OSMESA_INCLUDE_DIR})
    target_link_libraries(${CMAKE_PROJECT_NAME} ${OSMESA_LIBRARY})
else ()
    find_package(OpenGL REQUIRED)
    target_link_libraries(${CMAKE_PROJECT_NAME} ${OPENGL_LIBRARIES})
endif ()

target_link_libraries(${CMAKE_PROJECT_NAME}
                      ${3RDPARTY_LIBRARIES}
                      ${OMP_LIBRARIES})

# input_dirs: a list of absolute paths in the source dir
# output_dirs: a list of relative paths in the install dir
# input_base_dir: the base dir of input_dirs
function(SourcePath2InstallPath input_dirs output_dirs input_base_dir)
    get_filename_component(absolute_input_base_dir ${input_base_dir} ABSOLUTE)
    foreach(input_dir ${input_dirs})
        # We need to handle case where input_dir is not child subdirectory of
        # input_base_dir (or input_base_dir itself).
        # This could happen for pre-installed headers, e.g.
        #   input_dir == /usr/include/libpng16
        #   input_base_dir = /path-to-repo/Open3D
        if(${input_dir} MATCHES "^${absolute_input_base_dir}($|/.*)")
            # extract path relative to the project source dir
            set(relative_path "")
            file(RELATIVE_PATH relative_path ${absolute_input_base_dir} ${input_dir})

            # construct relative path in the install dir
            set(install_path "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}/${relative_path}")

            # save
            list(APPEND converted ${install_path})
        else()
            # When input_dir is not child subdirectory of input_base_dir,
            # we simply use the input_dir.
            list(APPEND converted ${input_dir})
        endif()
    endforeach()
    set (${output_dirs} ${converted} PARENT_SCOPE)
endfunction()

# build a list of include folders
SourcePath2InstallPath("${3RDPARTY_INCLUDE_DIRS_AT_INSTALL}" INSTALL_3RDPARTY_INCLUDE_DIRS ${PROJECT_SOURCE_DIR})

# set Open3D include directories
list(APPEND CONFIG_Open3D_INCLUDE_DIRS
    ${CMAKE_INSTALL_PREFIX}/include
    ${INSTALL_3RDPARTY_INCLUDE_DIRS}
)
list(REMOVE_DUPLICATES CONFIG_Open3D_INCLUDE_DIRS)


# set Open3D libraries
# If Open3D is a shared library
#     - Built-from-source 3rd-party libraries is embedded in Open3D
#     - Built-from-source 3rd-party libraries will NOT be installed
#     - A user app needs to link
#           1) Open3D
#           2) Pre-installed 3rd-party libraries
# If Open3D is a static library
#     - Built-from-source 3rd-party libraries is NOT embedded in Open3D
#     - Built-from-source 3rd-party libraries will be installed
#     - A user app needs to link
#           1) Open3D
#           2) Pre-installed 3rd-party libraries
#           3) Built-from-source 3rd-party libraries
list(APPEND CONFIG_Open3D_LIBRARIES "${CMAKE_PROJECT_NAME}" ${OMP_LIBRARIES})
if (BUILD_SHARED_LIBS)
    list(APPEND CONFIG_Open3D_LIBRARIES ${PRE_BUILT_3RDPARTY_LIBRARIES})
else ()
    list(APPEND CONFIG_Open3D_LIBRARIES ${3RDPARTY_LIBRARIES})
endif ()

# hot-fix for glfw vs glfw3 issue for Windows
# TODO: we shall revisit this for a better fix
# glfw is the target_name i.e. there is add_library(glfw) in glfw's cmake
# glfw3.lib (or libglfw3.a) is the library name of the compiled glfw
function(list_replace src_list dst_list src_val dst_val)
    foreach(val ${src_list})
        if(${val} STREQUAL ${src_val})
            message(STATUS "Replacing ${val} with ${dst_val}")
            list(APPEND converted ${dst_val})
        else()
            list(APPEND converted ${val})
        endif()
    endforeach()
    set(${dst_list} ${converted} PARENT_SCOPE)
endfunction(list_replace)

if (${BUILD_GLFW})
    message(STATUS "${CONFIG_Open3D_LIBRARIES}")
    list_replace("${CONFIG_Open3D_LIBRARIES}" CONFIG_Open3D_LIBRARIES "glfw" "glfw3")
    message(STATUS "${CONFIG_Open3D_LIBRARIES}")
endif()

# set Open3D library directories
list(APPEND CONFIG_Open3D_LIBRARY_DIRS
    "${CMAKE_INSTALL_PREFIX}/lib"
    ${3RDPARTY_LIBRARY_DIRS}
    ${OMP_LIBRARY_DIRS})

include(CMakePackageConfigHelpers)

# find_package Open3D
configure_package_config_file(Open3DConfig.cmake.in
                              "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/Open3DConfig.cmake"
                              INSTALL_DESTINATION ${INSTALL_CMAKE_DIR}
                              PATH_VARS CONFIG_Open3D_INCLUDE_DIRS CONFIG_Open3D_LIBRARY_DIRS
                              NO_SET_AND_CHECK_MACRO
                              NO_CHECK_REQUIRED_COMPONENTS_MACRO)

# find_package Open3D Version
write_basic_package_version_file("${PROJECT_BINARY_DIR}/Open3DConfigVersion.cmake"
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY ExactVersion)

# install
install(TARGETS ${CMAKE_PROJECT_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

file(GLOB HEADERS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS_ROOT} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME})

# Install the Open3DConfig.cmake and Open3DConfigVersion.cmake
install(FILES
        "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/Open3DConfig.cmake"
        "${PROJECT_BINARY_DIR}/Open3DConfigVersion.cmake"
        DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# uninstall target
if (NOT TARGET uninstall)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
                   IMMEDIATE @ONLY)

    add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P
                      ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif ()
