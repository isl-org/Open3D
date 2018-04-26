# build
file(GLOB_RECURSE IO_ALL_SOURCE_FILES "*.cpp")
file(GLOB         RPLY_SOURCE_FILES "../External/rply/*.c")
file(GLOB         LIBLZF_SOURCE_FILES "../External/liblzf/*.c")

include_directories(".")
include_directories("/ClassIO")
include_directories("/FileFormat")
include_directories(${JPEG_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIRS})
include_directories("../External/rply")
include_directories("../External/liblzf")
include_directories("../External/dirent")

add_library(IO
            ${IO_ALL_SOURCE_FILES}
            ${RPLY_SOURCE_FILES}
            ${LIBLZF_SOURCE_FILES})

target_link_libraries(IO
                      Core
                      tinyfiledialogs
                      ${JPEG_LIBRARIES}
                      ${PNG_LIBRARIES})

# install
install(TARGETS IO
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

file(GLOB HEADERS_ROOT     "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB HEADERS_CLASS_IO "${CMAKE_CURRENT_SOURCE_DIR}/ClassIO/*.h")

install(FILES ${HEADERS_ROOT}     DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Open3D/IO)
install(FILES ${HEADERS_CLASS_IO} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Open3D/IO/ClassIO)
