macro(ADD_OPEN3D_MODULE MODULE_NAME)
    file(GLOB MODULE_HEADER_FILES "${MODULE_NAME}/*.h")
    source_group("Header Files\\Open3D\\${PATH}" FILES ${MODULE_HEADER_FILES})
    file(GLOB MODULE_SOURCE_FILES "${MODULE_NAME}/*.cpp")
    source_group("Source Files\\Open3D\\${PATH}" FILES ${MODULE_SOURCE_FILES})
endmacro(ADD_OPEN3D_MODULE)

add_subdirectory(Core)
add_subdirectory(IO)
add_subdirectory(Visualization)
add_subdirectory(Tools)

# Unit tests
if (BUILD_UNIT_TESTS)
    add_subdirectory(UnitTest)
endif ()

# Python module
if (BUILD_PYTHON_MODULE)
    add_subdirectory(Python)
endif ()

# note: adding at least one real source file to any target that references
# reference: https://cmake.org/cmake/help/v3.12/command/add_library.html#object-libraries
add_library(${CMAKE_PROJECT_NAME} Open3DConfig.h Open3DConfig.cpp
            $<TARGET_OBJECTS:Core>
            $<TARGET_OBJECTS:IO>
            $<TARGET_OBJECTS:Visualization>)

# headless rendering
if (ENABLE_HEADLESS_RENDERING)
    find_package(OSMesa REQUIRED)
    include_directories(${OSMESA_INCLUDE_DIR})
    target_link_libraries(${CMAKE_PROJECT_NAME} ${OSMESA_LIBRARY})
else ()
    find_package(OpenGL REQUIRED)
    target_link_libraries(${CMAKE_PROJECT_NAME} ${OPENGL_LIBRARIES})
endif ()

# message(FATAL_ERROR ${3RDPARTY_LIBRARIES})
target_link_libraries(${CMAKE_PROJECT_NAME}
                      ${3RDPARTY_LIBRARIES}
                      ${OMP_LIBRARIES})

# input_dirs: a list of absolute paths in the source dir
# output_dirs: a list of absolute paths in the install dir
# input_base_dir: the base dir of input_dirs
function(SourcePath2InstallPath input_dirs output_dirs input_base_dir)
    foreach(input_dir ${input_dirs})
        # We need to handle case where input_dir is not child subdirectory of
        # input_base_dir. This could happen for pre-installed headers, e.g.
        #   input_dir == /usr/include/libpng16
        #   input_base_dir = /path-to-repo/Open3D
        if(${input_dir} MATCHES "^${input_base_dir}/.*")
            # extract path relative to the project source dir
            set(relative_path "")
            file(RELATIVE_PATH relative_path ${input_base_dir} ${input_dir})

            # construct absolute path in the install dir
            set(install_path "")
            get_filename_component(install_path
                                   "${relative_path}"
                                   REALPATH
                                   BASE_DIR "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}")

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
SourcePath2InstallPath("${CORE_INCLUDE_DIRS}"          INSTALL_CORE_INCLUDE_DIRS          ${CMAKE_CURRENT_SOURCE_DIR})
SourcePath2InstallPath("${IO_INCLUDE_DIRS}"            INSTALL_IO_INCLUDE_DIRS            ${CMAKE_CURRENT_SOURCE_DIR})
SourcePath2InstallPath("${VISUALIZATION_INCLUDE_DIRS}" INSTALL_VISUALIZATION_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
SourcePath2InstallPath("${3RDPARTY_INCLUDE_DIRS}"      INSTALL_3RDPARTY_INCLUDE_DIRS      ${CMAKE_CURRENT_SOURCE_DIR}/..)

# set Open3D include directories
list(APPEND CONFIG_Open3D_INCLUDE_DIRS
     "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}"
     ${INSTALL_CORE_INCLUDE_DIRS}
     ${INSTALL_IO_INCLUDE_DIRS}
     ${INSTALL_VISUALIZATION_INCLUDE_DIRS}
     ${INSTALL_3RDPARTY_INCLUDE_DIRS})
list(REMOVE_DUPLICATES CONFIG_Open3D_INCLUDE_DIRS)
list(SORT CONFIG_Open3D_INCLUDE_DIRS)

# set Open3D libraries
list(APPEND CONFIG_Open3D_LIBRARIES
     "${CMAKE_PROJECT_NAME}"
     ${3RDPARTY_LIBRARIES}
     ${OMP_LIBRARIES})

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

# find_package Open3D
configure_file(Open3DConfig.cmake.in
               "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/Open3DConfig.cmake" @ONLY)

# find_package Open3D Version
configure_file(Open3DConfigVersion.cmake.in
               "${PROJECT_BINARY_DIR}/Open3DConfigVersion.cmake" @ONLY)

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
