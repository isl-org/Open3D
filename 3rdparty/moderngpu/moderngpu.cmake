# Build system for header-only moderngpu libraries.
#

include(ExternalProject)


ExternalProject_Add(
    ext_moderngpu
    PREFIX moderngpu
    GIT_REPOSITORY https://github.com/moderngpu/moderngpu.git
    GIT_TAG 2b3985541c8e88a133769598c406c33ddde9d0a5
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_moderngpu SOURCE_DIR)
message(STATUS "moderngpu source dir: ${SOURCE_DIR}")

set(MODERNGPU_INCLUDE_DIRS "${SOURCE_DIR}/src/")
message(STATUS "moderngpu include dir: ${MODERNGPU_INCLUDE_DIRS}")