include_directories(${JPEG_INCLUDE_DIR} ${PNG_INCLUDE_DIRS})

file(GLOB IO_ALL_HEADER_FILES "*.h")
file(GLOB IO_ALL_SOURCE_FILES "*.cpp")
file(GLOB IO_RPLY_FILES "../External/rply/*.h" "../External/rply/*.c")
source_group("External\\RPly" FILES ${IO_RPLY_FILES})
file(GLOB IO_LIBLZF_FILES "../External/liblzf/*.h" "../External/liblzf/*.c")
source_group("External\\liblzf" FILES ${IO_LIBLZF_FILES})
file(GLOB IO_DIRENT_FILES "../External/dirent/*.h")
source_group("External\\dirent" FILES ${IO_DIRENT_FILES})
file(GLOB IO_CLASSIO_HEADER_FILES "ClassIO/*.h")
source_group("Header Files\\ClassIO" FILES ${IO_CLASSIO_HEADER_FILES})
file(GLOB IO_CLASSIO_SOURCE_FILES "ClassIO/*.cpp")
source_group("Source Files\\ClassIO" FILES ${IO_CLASSIO_SOURCE_FILES})
file(GLOB IO_FILEFORMAT_HEADER_FILES "FileFormat/*.h")
source_group("Header Files\\FileFormat" FILES ${IO_FILEFORMAT_HEADER_FILES})
file(GLOB IO_FILEFORMAT_SOURCE_FILES "FileFormat/*.cpp")
source_group("Source Files\\FileFormat" FILES ${IO_FILEFORMAT_SOURCE_FILES})

add_library(IO ${Open3D_LIB_TYPE}
    ${IO_ALL_HEADER_FILES} ${IO_ALL_SOURCE_FILES}
    ${IO_RPLY_FILES} ${IO_LIBLZF_FILES} ${IO_DIRENT_FILES}
    ${IO_CLASSIO_HEADER_FILES} ${IO_CLASSIO_SOURCE_FILES}
    ${IO_FILEFORMAT_HEADER_FILES} ${IO_FILEFORMAT_SOURCE_FILES}
)
target_link_libraries(IO Core tinyfiledialogs ${JPEG_LIBRARIES} ${PNG_LIBRARIES} ${JSONCPP_LIBRARIES})

# install(TARGETS IO
# 	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/Open3D
# 	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/Open3D
# 	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/Open3D
# )

# file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
# file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/ClassIO/*.h")
# file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/FileFormat/*.h")
# install(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/Open3D)

# INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#         DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D
#         FILES_MATCHING PATTERN "*.h*"
#         )

# # Only generate this target if no higher-level project already has
# if (NOT TARGET uninstall)
#     configure_file(cmake_uninstall.cmake.in
#                     cmake_uninstall.cmake IMMEDIATE @ONLY)

#     add_custom_target(uninstall
#                         "${CMAKE_COMMAND}" -P
#                         "${GLFW_BINARY_DIR}/cmake_uninstall.cmake")
#     set_target_properties(uninstall PROPERTIES FOLDER "GLFW3")
# endif()


# install(FILES "${CMAKE_SOURCE_DIR}/IO/IO.h"                                DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/IO)
# install(FILES "${CMAKE_SOURCE_DIR}/IO/ClassIO/PinholeCameraTrajectoryIO.h" DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/IO/ClassIO)
# install(FILES "${CMAKE_SOURCE_DIR}/IO/ClassIO/ImageIO.h"                   DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/IO/ClassIO)
# install(FILES "${CMAKE_SOURCE_DIR}/IO/ClassIO/PoseGraphIO.h"               DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/IO/ClassIO)
# install(FILES "${CMAKE_SOURCE_DIR}/IO/ClassIO/TriangleMeshIO.h"            DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/IO/ClassIO)
# install(FILES "${CMAKE_SOURCE_DIR}/IO/ClassIO/FeatureIO.h"                 DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/IO/ClassIO)
# install(FILES "${CMAKE_SOURCE_DIR}/IO/ClassIO/IJsonConvertible.h"          DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/IO/ClassIO)
# install(FILES "${CMAKE_SOURCE_DIR}/IO/ClassIO/IJsonConvertibleIO.h"        DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/IO/ClassIO)
# install(FILES "${CMAKE_SOURCE_DIR}/IO/ClassIO/PointCloudIO.h"              DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D/IO/ClassIO)

# INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/IO
#         DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D
#         FILES_MATCHING PATTERN "*.h*"
#         )

# INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/IO/ClassIO
#         DESTINATION ${CMAKE_INSTALL_PREFIX}include/Open3D
#         FILES_MATCHING PATTERN "*.h*"
#         )
