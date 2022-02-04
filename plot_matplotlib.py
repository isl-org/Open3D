from matplotlib import pyplot as plt
import open3d as o3d
import numpy as np
# import matplotlib
# matplotlib.use("TkAgg")


def main():
    pcd = o3d.io.read_point_cloud("examples/test_data/fragment.ply")
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors)
    plt.show()


if __name__ == "__main__":
    main()
    pass
