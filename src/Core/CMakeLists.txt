# build
file(GLOB_RECURSE CORE_ALL_SOURCE_FILES "*.cpp")

include_directories(".")
include_directories("/Camera")
include_directories("/Geometry")
include_directories("/Integration")
include_directories("/Odometry")
include_directories("/Registration")
include_directories("/Utility")

add_library(Core ${CORE_ALL_SOURCE_FILES})

target_link_libraries(Core ${JSONCPP_LIBRARIES})

# install
install(TARGETS Core
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

file(GLOB HEADERS_ROOT         "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB HEADERS_CAMERA       "${CMAKE_CURRENT_SOURCE_DIR}/Camera/*.h")
file(GLOB HEADERS_GEOMETRY     "${CMAKE_CURRENT_SOURCE_DIR}/Geometry/*.h")
file(GLOB HEADERS_INTEGRATION  "${CMAKE_CURRENT_SOURCE_DIR}/Integration/*.h")
file(GLOB HEADERS_ODOMETRY     "${CMAKE_CURRENT_SOURCE_DIR}/Odometry/*.h")
file(GLOB HEADERS_REGISTRATION "${CMAKE_CURRENT_SOURCE_DIR}/Registration/*.h")
file(GLOB HEADERS_UTILITY      "${CMAKE_CURRENT_SOURCE_DIR}/Utility/*.h")

install(FILES ${HEADERS_ROOT}         DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Open3D/Core)
install(FILES ${HEADERS_CAMERA}       DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Open3D/Core/Camera)
install(FILES ${HEADERS_GEOMETRY}     DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Open3D/Core/Geometry)
install(FILES ${HEADERS_INTEGRATION}  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Open3D/Core/Integration)
install(FILES ${HEADERS_ODOMETRY}     DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Open3D/Core/Odometry)
install(FILES ${HEADERS_REGISTRATION} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Open3D/Core/Registration)
install(FILES ${HEADERS_UTILITY}      DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Open3D/Core/Utility)
