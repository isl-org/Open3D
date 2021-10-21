target_sources(pybind PRIVATE
    blob.cpp
    core.cpp
    cuda_utils.cpp
    device.cpp
    dtype.cpp
    hashmap.cpp
    kernel.cpp
    linalg.cpp
    scalar.cpp
    size_vector.cpp
    tensor_accessor.cpp
    tensor_converter.cpp
    tensor.cpp
)

target_sources(pybind PRIVATE
    nns/nearest_neighbor_search.cpp
)
