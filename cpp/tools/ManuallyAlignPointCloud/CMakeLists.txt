add_executable(ManuallyAlignPointCloud)

target_sources(ManuallyAlignPointCloud PRIVATE
    ManuallyAlignPointCloud.cpp
    AlignmentSession.cpp
    VisualizerForAlignment.cpp
)
target_link_libraries(ManuallyAlignPointCloud PRIVATE
    Open3D::Open3D
    Open3D::3rdparty_jsoncpp
    Open3D::3rdparty_tinyfiledialogs
)

set_target_properties(ManuallyAlignPointCloud PROPERTIES FOLDER "Tools")
open3d_show_and_abort_on_warning(ManuallyAlignPointCloud)
open3d_set_global_properties(ManuallyAlignPointCloud)
