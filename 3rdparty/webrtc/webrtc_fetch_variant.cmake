# Cmake script to fetch and extract a single prebuilt WebRTC variant at build
# time by multi-config generators (Visual Studio).
#
# Required -D args:
#   URL    -- archive URL to download
#   SHA256 -- expected sha256 of the archive
#   DEST   -- extraction directory (recreated on a (re-)download)
#   STAMP  -- marker file created on success; if it already exists, this
#             script is a no-op.
foreach(var URL SHA256 DEST STAMP)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "webrtc_fetch_variant.cmake: -D${var}=... is required")
    endif()
endforeach()

if(EXISTS "${STAMP}")
    return()
endif()

file(REMOVE_RECURSE "${DEST}")
file(MAKE_DIRECTORY "${DEST}")

get_filename_component(archive_name "${URL}" NAME)
set(archive_path "${DEST}/../${archive_name}")

message(STATUS "Downloading prebuilt WebRTC: ${URL}")
file(DOWNLOAD "${URL}" "${archive_path}"
    EXPECTED_HASH SHA256=${SHA256}
    SHOW_PROGRESS
)
file(ARCHIVE_EXTRACT INPUT "${archive_path}" DESTINATION "${DEST}")
file(REMOVE "${archive_path}")
file(WRITE "${STAMP}" "ok")
