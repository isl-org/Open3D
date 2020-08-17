# build
file(GLOB_RECURSE GUI_SOURCE_FILES "*.cpp")
file(GLOB_RECURSE GUI_HEADER_FILES "*.h")

list(REMOVE_ITEM GUI_HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/Native.h)
if (WIN32)
    list(REMOVE_ITEM GUI_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/NativeLinux.cpp)
elseif (APPLE)
    list(REMOVE_ITEM GUI_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/NativeWin32.cpp)
    list(REMOVE_ITEM GUI_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/NativeLinux.cpp)
    list(APPEND GUI_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/NativeMacOS.mm)
    list(APPEND GUI_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/MenuMacOS.mm)
else ()
    list(REMOVE_ITEM GUI_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/NativeWin32.cpp)
endif()

# create object library
add_library(GUI OBJECT ${GUI_HEADER_FILES} ${GUI_SOURCE_FILES})
open3d_show_and_abort_on_warning(GUI)
open3d_set_global_properties(GUI)
open3d_link_3rdparty_libraries(GUI)

# --- build resources ----
file(GLOB GUI_RESOURCE_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/Resources/*)
file(GLOB GUI_MATERIAL_SOURCE_FILES "Materials/*.mat")

# copy GUI/Resources -> <output>/resources
set(GUI_RESOURCE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources)
file(MAKE_DIRECTORY ${GUI_RESOURCE_DIR})
file(COPY ${GUI_RESOURCE_SOURCE_FILES}
     DESTINATION ${GUI_RESOURCE_DIR})

# Compile .mat files and copy to resources directory in output directory
if (IOS OR ANDROID)
    set(GUI_MATC_ARGS --platform mobile)
elseif (NOT WIN32)
    set(GUI_MATC_ARGS --platform desktop)
endif()

set(GUI_MATERIAL_COMPILED_FILES "")
foreach(matsrc IN ITEMS ${GUI_MATERIAL_SOURCE_FILES})
    string(REGEX REPLACE "\\.[^.]*$" ".filamat" matdest ${matsrc})
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/Materials/" "${GUI_RESOURCE_DIR}/" matdest ${matdest})
    list(APPEND GUI_MATERIAL_COMPILED_FILES ${matdest})
    add_custom_command(OUTPUT ${matdest}
                       COMMAND ${FILAMENT_MATC} ${GUI_MATC_ARGS} -o ${matdest} ${matsrc}
		       MAIN_DEPENDENCY ${matsrc} DEPENDS ${FILAMENT_TARGET})
endforeach()

add_custom_target(GuiResources
                  DEPENDS ${GUI_MATERIAL_COMPILED_FILES})
add_dependencies(GUI GuiResources)

# Export GUI_RESOURCE_FILES to parent CMake context (cpp/open3d/)
set(GUI_RESOURCE_FILES
    ${GUI_RESOURCE_SOURCE_FILES} ${GUI_MATERIAL_COMPILED_FILES}
    PARENT_SCOPE)
set(GUI_RESOURCE_DIR ${GUI_RESOURCE_DIR} PARENT_SCOPE)

