include(ExternalProject)

set(ASSIMP_PATCH_FILES "${PROJECT_SOURCE_DIR}/3rdparty/assimp/ObjFileData.h")
list(APPEND ASSIMP_PATCH_FILES "${PROJECT_SOURCE_DIR}/3rdparty/assimp/ObjFileMtlImporter.cpp")
list(APPEND ASSIMP_PATCH_FILES "${PROJECT_SOURCE_DIR}/3rdparty/assimp/ObjFileImporter.cpp")

if(STATIC_WINDOWS_RUNTIME)
    set(ASSIMP_MSVC_RUNTIME "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    set(ASSIMP_MSVC_RUNTIME "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(ASSIMP_BUILD_TYPE "Release")
else()
    set(ASSIMP_BUILD_TYPE ${CMAKE_BUILD_TYPE})
endif()

ExternalProject_Add(
    ext_assimp
    PREFIX assimp
    GIT_REPOSITORY https://github.com/assimp/assimp
    GIT_TAG v5.0.1 # Jan 2020
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${ASSIMP_BUILD_TYPE}
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DASSIMP_NO_EXPORT=ON
        -DASSIMP_BUILD_ASSIMP_TOOLS=OFF
        -DASSIMP_BUILD_TESTS=OFF
        -DASSIMP_INSTALL_PDB=OFF
        -DASSIMP_BUILD_ZLIB=ON
        -DHUNTER_ENABLED=OFF # Renamed to "ASSIMP_HUNTER_ENABLED" in newer assimp.
	-DCMAKE_POLICY_DEFAULT_CMP0091=NEW
	-DCMAKE_MSVC_RUNTIME_LIBRARY=${ASSIMP_MSVC_RUNTIME}
	-DCMAKE_DEBUG_POSTFIX=
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
if (UNIX OR APPLE)
  set(ASSIMP_LIBRARIES assimp IrrXML)
else()
    set(ASSIMP_LIBRARIES assimp-vc142-mt IrrXML)
endif()
