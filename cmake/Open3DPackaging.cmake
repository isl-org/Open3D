# do this after all install(...) commands so that all targets are finalized.
# Essentially, the last thing included at the end of the top-level CMakeLists.txt
# https://www.scivision.dev/cmake-cpack-basic/

set(CPACK_GENERATOR TXZ)
if(WIN32)
    set(CPACK_GENERATOR ZIP)
endif()
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_VERSION ${OPEN3D_VERSION})
set(CPACK_PACKAGE_VENDOR "Open3D Team")
set(CPACK_PACKAGE_CONTACT "${PROJECT_EMAIL}")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package")
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
string(TOLOWER ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR} _sys)
string(TOLOWER ${PROJECT_NAME} _project_lower)
set(CPACK_PACKAGE_FILE_NAME
    "${_project_lower}-devel-${_sys}-${OPEN3D_VERSION_FULL}")
if (MSVC)
    set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_FILE_NAME}$<$<CONFIG:Debug>:-dbg>)
endif()

# not .gitignore as its regex syntax is distinct
# file(READ ${CMAKE_CURRENT_LIST_DIR}/.cpack_ignore _cpack_ignore)
# string(REGEX REPLACE "\n" ";" _cpack_ignore ${_cpack_ignore})
# set(CPACK_SOURCE_IGNORE_FILES "${_cpack_ignore}")

# install(FILES ${CPACK_RESOURCE_FILE_README} ${CPACK_RESOURCE_FILE_LICENSE}
#   DESTINATION share/docs/${PROJECT_NAME})

include(CPack)
