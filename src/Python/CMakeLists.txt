# Require Python.
find_package(PythonInterp REQUIRED)

set(PACKAGE_NAME open3d)

file(GLOB_RECURSE PY_ALL_HEADER_FILES "*.h")
file(GLOB_RECURSE PY_ALL_SOURCE_FILES "*.cpp")

ADD_SOURCE_GROUP(Core)
ADD_SOURCE_GROUP(IO)
ADD_SOURCE_GROUP(Tutorial)
ADD_SOURCE_GROUP(Visualization)
pybind11_add_module(${PACKAGE_NAME}
    ${PY_ALL_HEADER_FILES} ${PY_ALL_SOURCE_FILES}
    ${PY_CORE_SOURCE_FILES} ${PY_CORE_HEADER_FILES}
    ${PY_IO_SOURCE_FILES} ${PY_IO_HEADER_FILES}
    ${PY_VISUALIZATION_SOURCE_FILES} ${PY_VISUALIZATION_HEADER_FILES}
)

target_link_libraries(${PACKAGE_NAME} PRIVATE ${CMAKE_PROJECT_NAME})

if (BUILD_PYTHON_TUTORIALS)
    file(GLOB PY_BASIC_TUTORIAL_FILES "Tutorial/Basic/*.py")
    source_group("Basic" FILES ${PY_BASIC_TUTORIAL_FILES})
    file(GLOB PY_ADVANCED_TUTORIAL_FILES "Tutorial/Advanced/*.py")
    source_group("Advanced" FILES ${PY_ADVANCED_TUTORIAL_FILES})
    file(GLOB PY_APPLICATION_TUTORIAL_FILES "Tutorial/Application/*.py")
    source_group("Application" FILES ${PY_APPLICATION_TUTORIAL_FILES})
    file(GLOB PY_RECONSTRUCTION_SYSTEM_TUTORIAL_FILES "Tutorial/ReconstructionSystem/*.py")
    source_group("ReconstructionSystem" FILES ${PY_RECONSTRUCTION_SYSTEM_TUTORIAL_FILES})
    add_custom_target(PythonTutorial ALL SOURCES ${PY_BASIC_TUTORIAL_FILES} ${PY_ADVANCED_TUTORIAL_FILES} ${PY_APPLICATION_TUTORIAL_FILES} ${PY_RECONSTRUCTION_SYSTEM_TUTORIAL_FILES})
    set_target_properties(PythonTutorial PROPERTIES FOLDER "Python")
    add_custom_command(TARGET PythonTutorial POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/Python/Tutorial/" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/Tutorial")
    add_custom_command(TARGET PythonTutorial POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/Test/TestData/" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/TestData")
    add_dependencies(${PACKAGE_NAME} PythonTutorial)
endif (BUILD_PYTHON_TUTORIALS)

set_target_properties(${PACKAGE_NAME} PROPERTIES
                      FOLDER "Python"
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Python"
                      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/Python")

# find the python site-packages location
execute_process(COMMAND ${PYTHON_EXECUTABLE} -m site --user-site OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

# configure installer to copy the python module to the python site-packages
install(TARGETS ${PACKAGE_NAME} DESTINATION ${PYTHON_SITE_PACKAGES})
