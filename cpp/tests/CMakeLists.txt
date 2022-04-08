add_executable(tests)

target_sources(tests PRIVATE
    Main.cpp
)

target_link_libraries(tests PRIVATE
    Open3D
)

open3d_show_and_abort_on_warning(tests)
open3d_set_global_properties(tests)

