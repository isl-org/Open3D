# Raw URL link prefix of the "open3d_downloads" repo's master branch.
_repo_prefix = "https://github.com/intel-isl/open3d_downloads/raw/master"

# Map of URL to the download path relative to the download directory. The
# default download directory is Open3D/examples/test_data/open3d_downloads.
#
# For example:
# - relative download path: foo/bar/file.txt
# - file will be saved to : Open3D/examples/test_data/open3d_downloads/foo/bar/file.txt
#
# See https://github.com/intel-isl/open3d_downloads for details on how to
# manage the test data files.
map_url_to_relative_path = {
    f"{_repo_prefix}/RGBD/raycast_vtx_004.npy": "RGBD/raycast_vtx_004.npy",
    f"{_repo_prefix}/ICP/Civil.pcd": "ICP/Civil.pcd",
}
