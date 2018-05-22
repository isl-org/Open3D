# build
file(GLOB_RECURSE CORE_ALL_SOURCE_FILES "*.cpp")

include_directories(".")
include_directories("/Camera")
include_directories("/Geometry")
include_directories("/Integration")
include_directories("/Odometry")
include_directories("/Registration")
include_directories("/Utility")

# create object library
add_library(Core OBJECT ${CORE_ALL_SOURCE_FILES})

#install headers
file(GLOB HEADERS_ROOT         "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB HEADERS_CAMERA       "${CMAKE_CURRENT_SOURCE_DIR}/Camera/*.h")
file(GLOB HEADERS_GEOMETRY     "${CMAKE_CURRENT_SOURCE_DIR}/Geometry/*.h")
file(GLOB HEADERS_INTEGRATION  "${CMAKE_CURRENT_SOURCE_DIR}/Integration/*.h")
file(GLOB HEADERS_ODOMETRY     "${CMAKE_CURRENT_SOURCE_DIR}/Odometry/*.h")
file(GLOB HEADERS_REGISTRATION "${CMAKE_CURRENT_SOURCE_DIR}/Registration/*.h")
file(GLOB HEADERS_UTILITY      "${CMAKE_CURRENT_SOURCE_DIR}/Utility/*.h")

install(FILES ${HEADERS_ROOT}         DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}/Core)
install(FILES ${HEADERS_CAMERA}       DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}/Core/Camera)
install(FILES ${HEADERS_GEOMETRY}     DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}/Core/Geometry)
install(FILES ${HEADERS_INTEGRATION}  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}/Core/Integration)
install(FILES ${HEADERS_ODOMETRY}     DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}/Core/Odometry)
install(FILES ${HEADERS_REGISTRATION} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}/Core/Registration)
install(FILES ${HEADERS_UTILITY}      DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}/Core/Utility)
