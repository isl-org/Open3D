# Open3D 0.15 Release Notes

We are excited to bring you the best Open3D yet - version 0.15.

Starting from this release, we adopt a "tick-tock" model for balancing resolving issues vs. adding new features. In a nutshell, the "tick" releases are focused on resolving existing issues and eliminating bugs, while the "tock" releases mainly focus on developing new features. Open3D 0.15 is a "tick" release. We resolved [over 500 issues](https://github.com/isl-org/Open3D/issues?q=is%3Aissue+closed%3A2021-11-01..2022-02-15) for Open3D and Open3D-ML, as the infographic below illustrates.

![Issue_Stats](https://user-images.githubusercontent.com/93158890/154368347-6aa28948-a53a-4fc7-b1ea-de8a3c2501d7.png)

## Build System

* **[New]** We now provide Open3D binary packages for C++ users. No need to “Build from Source” - just download a binary package for your Operating System and use it “out of the box”. See [GitHub releases](https://github.com/isl-org/Open3D/releases/tag/v0.15) for v0.15 and [getting started guides](http://www.open3d.org/docs/latest/getting_started.html#c) for the latest development package.
* **[New]** Docker build tools to build ARM64 Linux wheels and multiple Python versions. See [ARM64 build guide](http://www.open3d.org/docs/release/arm.html) for more details.
* **[New]** Pre-compiled Open3D wheel for ARM64 Linux and macOS. Improved Apple Silicon support. Install Open3D on ARM64 Linux and macOS with `pip install open3d`.
* **[Update]** Open3D now builds with the new CXX11 ABI by default on Linux. Set `-DGLIBCXX_USE_CXX11_ABI=OFF` in `cmake` if you need the old ABI, e.g. to work with PyTorch / TensorFlow libraries.
* **[Update]** Starting with version 0.15, Open3D Conda packages are no longer supported. Install Open3D with `pip install open3d` inside a Conda virtual environment.

## Core

### Datasets

* **[New]** [Dataset module](http://www.open3d.org/docs/latest/tutorial/data/index.html) for automatically downloading and managing example data. The following example demonstrates how to create a `Dataset` object, extract its path, and display it in the Open3D Visualizer:
    ![Open3D 0.15 Dataset Demo](https://user-images.githubusercontent.com/1501945/154385188-600d701c-6c7d-4a7f-84bf-a7a60d87d6e6.gif)
    ```python
    import open3d as o3d

    if __name__ == "__main__":
        dataset = o3d.data.EaglePointCloud()
        pcd = o3d.io.read_point_cloud(dataset.path)
        o3d.visualization.draw(pcd)
    ```
    ```cpp
    #include <string>
    #include <memory>
    #include "open3d/Open3D.h"

    int main() {
        using namespace open3d;

        data::EaglePointCloud dataset;
        auto pcd = io::CreatePointCloudFromFile(dataset.GetPath());
        visualization::Draw({pcd});

        return 0;
    }
    ```

### Command Line Interface (CLI)

* **[New]** Open3D-dedicated Command Line Interface (CLI) for visualization and running Python examples. Below is a code snippet to get started with Open3D and its examples.
    ```sh
    # Install Open3D pip package
    pip install open3d

    # Print help
    open3d --help

    # List all runnable examples
    open3d example --list

    # Print source code of an example
    open3d example --show [category]/[example_name]

    # Run an example
    open3d example [category]/[example_name]

    # Run Open3D Viewer
    open3d draw

    # Open a mesh or point cloud file in Open3D Viewer
    open3d draw [filename]
    ```
    ![open3d_015_cli](https://user-images.githubusercontent.com/1501945/154750373-6a2cc8d9-c384-4d39-9726-1640c83ff387.gif)

* **[Update]** Python examples directory has been refactored for better namespace consistency and new examples have been added.

### Neighbor search
* **[Update]** Updated neighbor search module. The neighbor search module is now equipped with highly optimized built-in implementations of all search methods (Knn, Radius, and Hybrid) supporting both CPU and GPU. Faiss build dependency is removed.

## Visualization and GUI

* **[New]** Introducing raw mode visualization. The raw mode automatically simplifies the lighting environment and object materials to make it easier to inspect the underlying geometry of point clouds and triangle meshes.
    ![Open3D 0.15 Raw Mode Demo](https://user-images.githubusercontent.com/1501945/154383939-1fa65209-878c-4858-b18e-c1e4da7131b0.gif)
* **[New]** Open3D new visualizer now features CPU Rendering based on Mesa’s LLVMpipe software OpenGL implementation. Interactive applications, demos, and Python scripts are all supported as well as off-screen rendering. This feature can also be used within Docker.
    * Method 1: `LD_PRELOAD` from the command line
        ```bash
        LD_PRELOAD=/home/open3d/development/mesa-21.3.4/libGL.so python examples/python/visualization/draw.py
        ```
    * Method 2: Preload library in Python
        ```python
        import ctypes
        ctypes.cdll.LoadLibrary('/home/open3d/development/mesa-21.3.4/libGL.so')
        import open3d as o3d

        mesh = o3d.io.read_triangle_model('/home/open3d/development/FlightHelmet/FlightHelmet.gltf')
        o3d.visualization.draw(mesh)
        ```
    ![open3d_015_cpu_render](https://user-images.githubusercontent.com/1501945/154748967-cb46a1e7-2a97-4684-8dbb-42428cbffc3d.gif)

* **[New]** WidgetProxy and WidgetStack widgets allow the creation of user interfaces on the fly (contributed by @forrestjgq).
* **[New]** Background color option for button widgets (contributed by @forrestjgq).
* **[New]** Set maximum visible items in a list widget to prevent lists from growing too large (contributed by @forrestjgq).
* **[New]** Function keys are now bindable (contributed by @forrestjgq).
* **[New]** Support for specifying intrinsic projection matrix in the new visualizer.
* **[New]** Add support for scaling 3D labels.
* **[Fix]** Open3D for TensorBoard plugin does not need Open3D-ML now.
* **[Fix]** Point picking, text input, and UI layout (contributed by @forrestjgq).

## Geometry

* **[Fix]** Oriented Bounding Box
    * Fixed an issue where the orientation of the `OrientedBoundingBox` was mirrored.
    * **[New]** added a new parameter for robust oriented bounding box computation for degenerated point clouds.
* **[Fix]** Convex hull meshes created from point clouds now have outward-pointing triangles.
* **[Update]** Added a new parameter for robust convex hull computation.
* **[Update]** TriangleMesh `GetSelfIntersectingTriangles()` and related functions like `IsWatertight()`, `GetVolume()`, etc. are now more than 4 times faster.
* **[Fix]** Corrected an issue with `io::AddTrianglesByEarClipping()` where the algorithm could fail for concave polygons.
* **[New]** New Python examples for reconstruction and voxelization.
* **[Fix]** Improved logger performance.

## Open3D-ML

* **[New]** MIT-licensed implementation of RandLANet.
    ![open3D_015_randlanet](https://user-images.githubusercontent.com/1501945/154385894-a7845b7d-4890-4cea-9032-e9f5fdc038a3.gif)
* **[New]** Intel OpenVINO inference backend (contributed by @dkurt).
* **[Fix]** Fixed an issue with S3DIS where the loss gets NaN after a few epochs.
* **[Fix]** Fixed an issue with IoU calculation which fails for large point clouds while running inference in patches.
* **[Fix]** Fixed an issue where the labels are not correctly ordered in the visualizer.
* **[New]** Support for Images in Dataset Visualizer (contributed by @ajinkyakhoche).
     ![open3D_015_ml_image_vis](https://user-images.githubusercontent.com/1501945/154386380-5b5df7b7-b5b9-4d79-849a-359baadd37e3.gif)


## **Acknowledgment**

We would like to thank all of our community contributors for their true labor of love for this release!

@ajinkyakhoche @ceroytres @chunibyo-wly @dkurt @forrestjgq @Fuhrmann-sep @jeertmans @junha-l @mag-sruehl @maxim0815 @Nicholas-Mitchell @nigels-com @NobuoTsukamoto @ntw-au @roehling @theNded

Also thanks to the many others who helped the Open3D community by reporting as well as resolving issues.
