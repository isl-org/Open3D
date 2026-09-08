include(ExternalProject)

find_package(Git QUIET REQUIRED)

if(MSVC)
    set(lib_name assimp-vc${MSVC_TOOLSET_VERSION}-mt)
else()
    set(lib_name assimp)
endif()

# IntelLLVM (SYCL) compiler defaults to fast math, causing NaN comparison code
# compilation error.
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    set(assimp_cmake_cxx_flags "${CMAKE_CXX_FLAGS} -ffp-contract=on -fno-fast-math")
else()
    set(assimp_cmake_cxx_flags "${CMAKE_CXX_FLAGS}")
endif()

# Assimp import/export toggles aligned with Open3D mesh I/O (see TriangleMeshIO and
# FileASSIMP). Native Open3D readers/writers handle ply and legacy off read; assimp
# is enabled only for the formats below.
#
# Importers: obj, stl, off, gltf/glb, fbx, usd
# Exporters: gltf/glb (embedded textures), obj, stl, fbx (best-effort geometry)

ExternalProject_Add(
    ext_assimp
    PREFIX assimp
    URL https://github.com/assimp/assimp/archive/refs/tags/v6.0.5.zip
    URL_HASH SHA256=89e477e89df739ecc7dd21874897ac2e70c8944eb1074fbdbbdbf4dd2de6a403
    DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/assimp"
    UPDATE_COMMAND ""
    PATCH_COMMAND ${GIT_EXECUTABLE} init
    COMMAND ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace
        ${CMAKE_CURRENT_LIST_DIR}/0001-fix-usd-AddProperty-aiString-pointer.patch
    COMMAND ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace
        ${CMAKE_CURRENT_LIST_DIR}/0002-fix-usd-embedded-texture-mwidth-bytes.patch
    COMMAND ${GIT_EXECUTABLE} apply --ignore-space-change --ignore-whitespace
        ${CMAKE_CURRENT_LIST_DIR}/0003-fix-draco-cmake-intelllvm-cxx-flags.patch
    CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS_hidden}
        -DCMAKE_CXX_FLAGS:STRING=${assimp_cmake_cxx_flags}
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DASSIMP_BUILD_ASSIMP_TOOLS=OFF
        -DASSIMP_BUILD_TESTS=OFF
        -DASSIMP_INSTALL_PDB=OFF
        -DASSIMP_BUILD_ZLIB=OFF
        -DZLIB_ROOT=${CMAKE_BINARY_DIR}/zlib
        -DASSIMP_NO_EXPORT=OFF
        -DASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT=OFF
        -DASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT=OFF
        -DASSIMP_HUNTER_ENABLED=OFF
        -DASSIMP_BUILD_DRACO=ON
        -DASSIMP_BUILD_DRACO_STATIC=ON
        -DASSIMP_WARNINGS_AS_ERRORS=OFF
        -DCMAKE_DEBUG_POSTFIX=
        # Mesh I/O via FileASSIMP / TriangleMeshIO (see assimp.cmake header comment)
        -DASSIMP_BUILD_OBJ_IMPORTER=ON
        -DASSIMP_BUILD_STL_IMPORTER=ON
        -DASSIMP_BUILD_OFF_IMPORTER=ON
        -DASSIMP_BUILD_GLTF_IMPORTER=ON
        -DASSIMP_BUILD_FBX_IMPORTER=ON
        -DASSIMP_BUILD_USD_IMPORTER=ON   # USD (tinyusdz)
        -DASSIMP_BUILD_GLTF_EXPORTER=ON
        -DASSIMP_BUILD_OBJ_EXPORTER=ON
        -DASSIMP_BUILD_STL_EXPORTER=ON
        -DASSIMP_BUILD_FBX_EXPORTER=ON
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}IrrXML${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${Open3D_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}draco${CMAKE_STATIC_LIBRARY_SUFFIX}
)

ExternalProject_Get_Property(ext_assimp INSTALL_DIR)
set(ASSIMP_INCLUDE_DIR ${INSTALL_DIR}/include/)
set(ASSIMP_LIB_DIR ${INSTALL_DIR}/${Open3D_INSTALL_LIB_DIR})
set(ASSIMP_LIBRARIES ${lib_name} draco)
