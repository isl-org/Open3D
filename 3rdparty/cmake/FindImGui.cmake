find_path(ImGui_INCLUDE_DIR NAMES imgui.h PATH_SUFFIXES imgui)
find_library(ImGui_LIBRARY NAMES imgui)
find_library(stb_LIBRARY NAMES stb)
if(ImGui_INCLUDE_DIR AND ImGui_LIBRARY)
    add_library(ImGui::ImGui UNKNOWN IMPORTED)
    set_target_properties(ImGui::ImGui PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ImGui_INCLUDE_DIR}"
        IMPORTED_LOCATION "${ImGui_LIBRARY}"
    )
    if(stb_LIBRARY)
        set_target_properties(ImGui::ImGui PROPERTIES
            INTERFACE_LINK_LIBRARIES "${stb_LIBRARY}"
        )
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ImGui  DEFAULT_MSG  ImGui_LIBRARY ImGui_INCLUDE_DIR)

