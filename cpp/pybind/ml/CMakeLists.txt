target_sources(pybind PRIVATE
    ml.cpp
)

target_sources(pybind PRIVATE
    contrib/contrib_subsample.cpp
    contrib/contrib.cpp
)

target_sources(pybind PRIVATE
    contrib/iou.cpp
)
# iou.cpp instantiates a real sycl::queue (via SYCLContext::GetDefaultQueue) to
# call IoU{Bev,3d}SYCLKernel, so it must itself be compiled with -fsycl. Source
# file properties are directory-scoped in CMake and are not visible to a
# target's compile rules unless set in the same directory the target was
# created in (or a parent), so this flag is set on `pybind` (created in
# ../CMakeLists.txt) from there, not here.
