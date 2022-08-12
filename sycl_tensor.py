import open3d as o3d
import open3d.core as o3c
import numpy as np
import time


def binary_ew():
    a = o3c.Tensor([1, 1], dtype=o3c.float32, device=o3c.Device("SYCL:0"))
    b = o3c.Tensor([2, 2], dtype=o3c.float32, device=o3c.Device("SYCL:0"))
    c = a + b
    print(c)


def advanced_indexing():
    device = o3c.Device("SYCL:0")

    np_src = np.array(range(24)).reshape((2, 3, 4))
    o3_src = o3c.Tensor(np_src, device=device)

    np_fill = np.array(([[100, 200], [300, 400]]))
    o3_fill = o3c.Tensor(np_fill, device=device)

    np_src[1, 0:2, [1, 2]] = np_fill
    o3_src[1, 0:2, [1, 2]] = o3_fill
    np.testing.assert_equal(o3_src.cpu().numpy(), np_src)


def advanced_indexing_bench():
    repeat = 50

    full_dim = 10_000_000
    select_dim = int(full_dim / 2)

    # Numpy
    np_src = np.zeros((full_dim, 3), dtype=np.int32)
    np_indices = np.random.randint(0, full_dim, size=select_dim)
    np_src[np_indices, :] = 1
    np_src[np_indices, :] = 1

    start_time = time.time()
    for _ in range(repeat):
        np_src[np_indices, :] = 1
    end_time = time.time()
    np_time = (end_time - start_time) / repeat
    print(f"Numpy time: {np_time:.4f}s")

    # CPU.
    cpu_device = o3c.Device("CPU:0")
    cpu_src = o3c.Tensor(np_src, device=cpu_device)
    cpu_indices = o3c.Tensor(np_indices, device=cpu_device)
    cpu_src[cpu_indices, :] = 1
    cpu_src[cpu_indices, :] = 1

    start_time = time.time()
    for _ in range(repeat):
        cpu_src[cpu_indices, :] = 1
    end_time = time.time()
    cpu_time = (end_time - start_time) / repeat
    print(f"CPU time: {cpu_time:.4f}s")

    # SYCL.
    sycl_device = o3c.Device("SYCL:0")
    sycl_src = o3c.Tensor(np_src, device=sycl_device)
    sycl_indices = o3c.Tensor(np_indices, device=sycl_device)
    sycl_src[sycl_indices, :] = 1
    sycl_src[sycl_indices, :] = 1

    start_time = time.time()
    for _ in range(repeat):
        sycl_src[sycl_indices, :] = 1
    end_time = time.time()
    sycl_time = (end_time - start_time) / repeat
    print(f"SYCL time: {sycl_time:.4f}s")


if __name__ == "__main__":
    advanced_indexing()
    advanced_indexing_bench()
