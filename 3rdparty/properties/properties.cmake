include(ExternalProject)

set(ASSIMP_PATCH_FILES "${PROJECT_SOURCE_DIR}/3rdparty/assimp/ObjFileData.h")
list(APPEND ASSIMP_PATCH_FILES "${PROJECT_SOURCE_DIR}/3rdparty/assimp/ObjFileMtlImporter.cpp")
list(APPEND ASSIMP_PATCH_FILES "${PROJECT_SOURCE_DIR}/3rdparty/assimp/ObjFileImporter.cpp")

if(STATIC_WINDOWS_RUNTIME)
    set(ASSIMP_MSVC_RUNTIME "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    set(ASSIMP_MSVC_RUNTIME "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
ExternalProject_Add(
    ext_properties
    PREFIX properties
    GIT_REPOSITORY https://gitlab.com/LIONant/properties.git
    UPDATE_COMMAND ""
)

ExternalProject_Get_Property(ext_properties INSTALL_DIR)
ExternalProject_Get_Property(ext_properties SOURCE_DIR)
# ExternalProject_Add_Step(ext_assimp patch-copy
#   COMMAND ${CMAKE_COMMAND} -E copy ${ASSIMP_PATCH_FILES} ${SOURCE_DIR}/code/Obj
#   COMMAND ${CMAKE_COMMAND} -E echo "Copying patch files for Obj loader into assimp source"
#   DEPENDEES download
#   DEPENDERS update)
# set(ASSIMP_INCLUDE_DIR ${INSTALL_DIR}/include/)
# set(ASSIMP_LIB_DIR ${INSTALL_DIR}/lib)
# set(ASSIMP_LIBRARIES ${lib_name} IrrXML)
