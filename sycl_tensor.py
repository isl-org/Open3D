import open3d as o3d
import open3d.core as o3c


def main():
    a = o3c.Tensor([1, 1], dtype=o3c.float32, device=o3c.Device("SYCL:0"))
    b = o3c.Tensor([2, 2], dtype=o3c.float32, device=o3c.Device("SYCL:0"))
    c = a + b
    print(c)


if __name__ == "__main__":
    main()
