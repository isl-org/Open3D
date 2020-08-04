include(ExternalProject)

ExternalProject_Add(
    ext_assimp
    PREFIX assimp
    GIT_REPOSITORY https://github.com/assimp/assimp
    GIT_TAG v5.0.1 # Jan 2020
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DASSIMP_NO_EXPORT=ON
        -DASSIMP_BUILD_ASSIMP_TOOLS=OFF
        -DASSIMP_BUILD_TESTS=OFF
)

ExternalProject_Get_Property(ext_assimp INSTALL_DIR)
message("INSTALL_DIR of myExtProj = ${INSTALL_DIR}")

set(ASSIMP_INCLUDE_DIR ${INSTALL_DIR}/include/)
set(ASSIMP_LIB_DIR ${INSTALL_DIR}/lib)
set(ASSIMP_LIBRARIES assimp IrrXML)
