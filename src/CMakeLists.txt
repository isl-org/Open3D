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

target_link_libraries(${CMAKE_PROJECT_NAME}
                      ${3RDPARTY_LIBRARIES}
                      ${OMP_LIBRARIES})

# input:  a list of absolute paths in the current source dir
# output: a list of absolute paths in the install dir
function(SourcePath2InstallPath input output)
    foreach(child ${input})

        set(relative_path "")
        set(install_path "")

        # extract path relative to the project source dir
        file(RELATIVE_PATH relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${child})

        # construct absolute path in the install dir
        get_filename_component(install_path "${relative_path}"
                               REALPATH
                               BASE_DIR "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}")

        # save
        list(APPEND converted ${install_path})
    endforeach()
    set (${output} ${converted} PARENT_SCOPE)
endfunction()

# build a list of include folders
SourcePath2InstallPath("${CORE_INCLUDE_DIRS}"          INSTALL_CORE_INCLUDE_DIRS)
SourcePath2InstallPath("${IO_INCLUDE_DIRS}"            INSTALL_IO_INCLUDE_DIRS)
SourcePath2InstallPath("${VISUALIZATION_INCLUDE_DIRS}" INSTALL_VISUALIZATION_INCLUDE_DIRS)
SourcePath2InstallPath("${3RDPARTY_INCLUDE_DIRS}"      INSTALL_3RDPARTY_INCLUDE_DIRS)

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
