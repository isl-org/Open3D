include(ExternalProject)

ExternalProject_Add(
    ext_moderngpu
    PREFIX moderngpu
    URL https://github.com/moderngpu/moderngpu/archive/2b3985541c8e88a133769598c406c33ddde9d0a5.zip
    URL_HASH SHA256=191546af18cd5fb858ecb561316f3af67537ab16f610fc8f1a5febbffc27755a
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/moderngpu"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_moderngpu SOURCE_DIR)
message(STATUS "moderngpu source dir: ${SOURCE_DIR}")

set(MODERNGPU_INCLUDE_DIRS "${SOURCE_DIR}/src/")
message(STATUS "moderngpu include dir: ${MODERNGPU_INCLUDE_DIRS}")

