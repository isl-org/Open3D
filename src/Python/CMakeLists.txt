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

set_target_properties(${PACKAGE_NAME} PROPERTIES
                      FOLDER "Python"
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Python"
                      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}/Python")

# find the python site-packages location
execute_process(COMMAND ${PYTHON_EXECUTABLE} -m site --user-site OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)

# configure installer to copy the python module to the python site-packages
install(TARGETS ${PACKAGE_NAME} DESTINATION ${PYTHON_SITE_PACKAGES})
