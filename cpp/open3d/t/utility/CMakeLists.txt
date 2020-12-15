# Build
set(SRC
    SolveTransformation.cpp
)

# Create object library
add_library(tutility OBJECT
            ${SRC})
open3d_show_and_abort_on_warning(tutility)
open3d_set_global_properties(tutility)
open3d_set_open3d_lib_properties(tutility)
open3d_link_3rdparty_libraries(tutility)

