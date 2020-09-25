# Build system for header-only boost libraries.
#
# In general, we prefer avoiding boost or use header-only boost libraries.
# Compiling boost libraries can addup to the build time.
#
# Current boost libraries:
# - predef (header-only)

include(ExternalProject)

if(WIN32)
    message(FATAL_ERROR "Win32 not supported.")
endif()

ExternalProject_Add(
    ext_boost
    PREFIX boost
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_TAG boost-1.73.0
    GIT_SUBMODULES tools/boostdep libs/predef # Only need a subset of boost
    GIT_SHALLOW ON                            # git clone --depth 1
    GIT_SUBMODULES_RECURSE OFF
    BUILD_IN_SOURCE ON
    CONFIGURE_COMMAND ""
    BUILD_COMMAND echo "Running Boost build..."
    COMMAND python tools/boostdep/depinst/depinst.py predef
    COMMAND ./bootstrap.sh
    COMMAND ./b2 headers
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_boost SOURCE_DIR)
message(STATUS "Boost source dir: ${SOURCE_DIR}")

# By default, BOOST_INCLUDE_DIRS should not have trailing "/".
# The actual headers files are located in `${SOURCE_DIR}/boost`.
set(BOOST_INCLUDE_DIRS ${SOURCE_DIR}/ext_boost)
