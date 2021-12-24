function(open3d_generate_license target_file)
    # Only write to ${target_file} if it doesn't exist or if it's different to
    # avoid recompilation.
    function(write_file_if_different target_file value)
        if(EXISTS "${target_file}")
            file(READ ${target_file} existing_value)
            if("${value}" STREQUAL "${existing_value}")
                string(LENGTH "${existing_value}" str_len)
                message(STATUS "No changes to ${target_file}, file lenth ${str_len} !!!!!!!!!!!!!!!!!!!!!")
            else()
                message(STATUS "Updating ${target_file}")
                file(WRITE "${target_file}" "${value}")
            endif()
        else()
            message(STATUS "Creating ${target_file} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            file(WRITE "${target_file}" "${value}")
        endif()
    endfunction()

    string(REPEAT "#" 80 separator)

    # Third party license
    file(GLOB license_files
        "${PROJECT_SOURCE_DIR}/3rdparty/*/LICENSE")
    set(third_party_header "constexpr char s_generated_third_party_license[] = R\"open3d_delimiter\(")
    set(third_party_footer "\)open3d_delimiter\";\n")
    set(third_party_license "")
    foreach(license_file IN LISTS license_files)
        get_filename_component(license_dir ${license_file} DIRECTORY)
        get_filename_component(library_name ${license_dir} NAME)
        file(READ ${license_file} license_str)
        set(third_party_license "${third_party_license}${separator}\n")
        set(third_party_license "${third_party_license}# License for ${library_name}\n")
        set(third_party_license "${third_party_license}${separator}\n")
        set(third_party_license "${third_party_license}${license_str}\n\n")
    endforeach()
    set(third_party_license "${third_party_header}${third_party_license}${third_party_footer}")

    # Open3D license
    set(open3d_header "constexpr char s_generated_open3d_license[] = R\"open3d_delimiter\(")
    set(open3d_footer "\)open3d_delimiter\";\n")
    file(READ "${PROJECT_SOURCE_DIR}/LICENSE" license_str)
    set(open3d_license "${open3d_header}${license_str}${open3d_footer}")

    set(all_license "${third_party_license}\n${open3d_license}")
    write_file_if_different("${target_file}" "${all_license}")
endfunction()
