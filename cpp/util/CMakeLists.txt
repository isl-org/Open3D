add_executable(EmbedResources)

target_sources(EmbedResources PRIVATE EmbedResources.cpp)
target_link_libraries(EmbedResources PRIVATE Open3D::Open3D)

set_target_properties(EmbedResources PROPERTIES FOLDER "Util")
open3d_show_and_abort_on_warning(EmbedResources)
open3d_set_global_properties(EmbedResources)
