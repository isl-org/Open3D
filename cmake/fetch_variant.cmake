# Fetch and extract one SHA256-verified prebuilt artifact variant.
#
# Required -D args: URL, SHA256, DEST, STAMP.
# Optional -D arg: CACHE_DIR. When omitted, the archive is temporary and is
# removed after extraction. A cache is useful for config-specific downloads.
foreach(var URL SHA256 DEST STAMP)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "fetch_variant.cmake: -D${var}=... is required")
    endif()
endforeach()

if(EXISTS "${STAMP}")
    return()
endif()

get_filename_component(archive_name "${URL}" NAME)
if(CACHE_DIR)
    file(MAKE_DIRECTORY "${CACHE_DIR}")
    set(archive_path "${CACHE_DIR}/${archive_name}")
else()
    get_filename_component(dest_parent "${DEST}" DIRECTORY)
    file(MAKE_DIRECTORY "${dest_parent}")
    set(archive_path "${dest_parent}/${archive_name}")
endif()

if(EXISTS "${archive_path}")
    file(SHA256 "${archive_path}" archive_sha256)
    if(NOT archive_sha256 STREQUAL SHA256)
        file(REMOVE "${archive_path}")
    endif()
endif()

if(NOT EXISTS "${archive_path}")
    message(STATUS "Downloading prebuilt artifact: ${URL}")
    file(DOWNLOAD "${URL}" "${archive_path}"
        EXPECTED_HASH SHA256=${SHA256}
        SHOW_PROGRESS)
endif()

file(REMOVE_RECURSE "${DEST}")
file(MAKE_DIRECTORY "${DEST}")
file(ARCHIVE_EXTRACT INPUT "${archive_path}" DESTINATION "${DEST}")
if(NOT CACHE_DIR)
    file(REMOVE "${archive_path}")
endif()
get_filename_component(stamp_dir "${STAMP}" DIRECTORY)
file(MAKE_DIRECTORY "${stamp_dir}")
file(WRITE "${STAMP}" "ok")