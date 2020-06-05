import open3d as o3d


def test_global_flags():
    assert o3d.open3d_pybind._GLIBCXX_USE_CXX11_ABI in (True, False)
    assert o3d.open3d_pybind._ENABLE_HEADLESS_RENDERING in (True, False)
    assert o3d.open3d_pybind._BUILD_CUDA_MODULE in (True, False)
