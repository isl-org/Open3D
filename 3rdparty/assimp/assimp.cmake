include(ExternalProject)

set(ASSIMP_PATCH_FILES "${PROJECT_SOURCE_DIR}/3rdparty/assimp/ObjFileData.h")
list(APPEND ASSIMP_PATCH_FILES "${PROJECT_SOURCE_DIR}/3rdparty/assimp/ObjFileMtlImporter.cpp")
list(APPEND ASSIMP_PATCH_FILES "${PROJECT_SOURCE_DIR}/3rdparty/assimp/ObjFileImporter.cpp")

if(MSVC)
    set(lib_name assimp-vc142-mt)
else()
    set(lib_name assimp)
endif()

ExternalProject_Add(
    ext_assimp
    PREFIX assimp
    URL https://github.com/assimp/assimp/archive/refs/tags/v5.0.1.tar.gz
    URL_HASH SHA256=11310ec1f2ad2cd46b95ba88faca8f7aaa1efe9aa12605c55e3de2b977b3dbfc
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/assimp"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DASSIMP_NO_EXPORT=ON
        -DASSIMP_BUILD_ASSIMP_TOOLS=OFF
        -DASSIMP_BUILD_TESTS=OFF
        -DASSIMP_INSTALL_PDB=OFF
        -DASSIMP_BUILD_ZLIB=ON
        -DHUNTER_ENABLED=OFF # Renamed to "ASSIMP_HUNTER_ENABLED" in newer assimp.
        -DCMAKE_DEBUG_POSTFIX=
        ${ExternalProject_CMAKE_ARGS_hidden}
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}IrrXML${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_assimp INSTALL_DIR)
ExternalProject_Get_Property(ext_assimp SOURCE_DIR)
ExternalProject_Add_Step(ext_assimp patch-copy
  COMMAND ${CMAKE_COMMAND} -E copy ${ASSIMP_PATCH_FILES} ${SOURCE_DIR}/code/Obj
  COMMAND ${CMAKE_COMMAND} -E echo "Copying patch files for Obj loader into assimp source"
  DEPENDEES download
  DEPENDERS update)
set(ASSIMP_INCLUDE_DIR ${INSTALL_DIR}/include/)
set(ASSIMP_LIB_DIR ${INSTALL_DIR}/lib)
set(ASSIMP_LIBRARIES ${lib_name} IrrXML)
