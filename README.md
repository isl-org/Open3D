<p align="center">
<img src="https://raw.githubusercontent.com/isl-org/Open3D/master/docs/_static/open3d_logo_horizontal.png" width="320" />
</p>

# Open3D: A Modern Library for 3D Data Processing

<h4>
    <a href="http://www.open3d.org">Homepage</a> |
    <a href="http://www.open3d.org/docs">Docs</a> |
    <a href="http://www.open3d.org/docs/release/getting_started.html">Quick Start</a> |
    <a href="http://www.open3d.org/docs/release/compilation.html">Compile</a> |
    <a href="http://www.open3d.org/docs/release/index.html#python-api-index">Python</a> |
    <a href="http://www.open3d.org/docs/release/cpp_api.html">C++</a> |
    <a href="https://github.com/isl-org/Open3D-ML">Open3D-ML</a> |
    <a href="https://github.com/isl-org/Open3D/releases">Viewer</a> |
    <a href="http://www.open3d.org/docs/release/contribute/contribute.html">Contribute</a> |
    <a href="https://www.youtube.com/channel/UCRJBlASPfPBtPXJSPffJV-w">Demo</a> |
    <a href="https://github.com/isl-org/Open3D/discussions">Forum</a>
</h4>

Open3D is an open-source library that supports rapid development of software
that deals with 3D data. The Open3D frontend exposes a set of carefully selected
data structures and algorithms in both C++ and Python. The backend is highly
optimized and is set up for parallelization. We welcome contributions from
the open-source community.

[![Ubuntu CI](https://github.com/isl-org/Open3D/workflows/Ubuntu%20CI/badge.svg)](https://github.com/isl-org/Open3D/actions?query=workflow%3A%22Ubuntu+CI%22)
[![macOS CI](https://github.com/isl-org/Open3D/workflows/macOS%20CI/badge.svg)](https://github.com/isl-org/Open3D/actions?query=workflow%3A%22macOS+CI%22)
[![Windows CI](https://github.com/isl-org/Open3D/workflows/Windows%20CI/badge.svg)](https://github.com/isl-org/Open3D/actions?query=workflow%3A%22Windows+CI%22)

**Core features of Open3D include:**

* 3D data structures
* 3D data processing algorithms
* Scene reconstruction
* Surface alignment
* 3D visualization
* Physically based rendering (PBR)
* 3D machine learning support with PyTorch and TensorFlow
* GPU acceleration for core 3D operations
* Available in C++ and Python

For more, please visit the [Open3D documentation](http://www.open3d.org/docs).

## Python quick start

Pre-built pip and conda packages support Ubuntu 18.04+, macOS 10.15+ and
Windows 10 (64-bit) with Python 3.6-3.9.

```bash
# Install Open3D stable release with pip (including in conda virtual environments)
$ pip install open3d

# Test the installation
$ python -c "import open3d as o3d; print(o3d)"

```

To get the latest features in Open3D, install the
[development pip package](http://www.open3d.org/docs/latest/getting_started.html#development-version-pip).
To compile Open3D from source, refer to
[compiling from source](http://www.open3d.org/docs/release/compilation.html).

## C++ quick start

Checkout the following links to get started with Open3D C++ API

* Download Open3D binary package: [Release](https://github.com/isl-org/Open3D/releases) or [latest development version](http://www.open3d.org/docs/latest/getting_started.html#c)
* [Compiling Open3D from source](http://www.open3d.org/docs/release/compilation.html)
* [Open3D C++ API](http://www.open3d.org/docs/release/cpp_api.html)

To use Open3D in your C++ project, checkout the following examples

* [Find Pre-Installed Open3D Package in CMake](https://github.com/isl-org/open3d-cmake-find-package)
* [Use Open3D as a CMake External Project](https://github.com/isl-org/open3d-cmake-external-project)

## Open3D-Viewer app

<img width="480" src="https://raw.githubusercontent.com/isl-org/Open3D/master/docs/_static/open3d_viewer.png">

Open3D-Viewer is a standalone 3D viewer app available on Ubuntu and macOS.
Please stay tuned for Windows. Download Open3D Viewer from the
[release page](https://github.com/isl-org/Open3D/releases).

## Open3D-ML

<img width="480" src="https://raw.githubusercontent.com/isl-org/Open3D-ML/master/docs/images/getting_started_ml_visualizer.gif">

Open3D-ML is an extension of Open3D for 3D machine learning tasks. It builds on
top of the Open3D core library and extends it with machine learning tools for
3D data processing. To try it out, install Open3D with PyTorch or TensorFlow and check out
[Open3D-ML](https://github.com/isl-org/Open3D-ML).

## Communication channels

* [GitHub Issue](https://github.com/isl-org/Open3D/issues): bug reports,
  feature requests, etc.
* [Forum](https://github.com/isl-org/Open3D/discussions): discussion on the usage of Open3D.
* [Discord Chat](https://discord.gg/D35BGvn): online chats, discussions,
  and collaboration with other users and developers.

## Citation

Please cite [our work](https://arxiv.org/abs/1801.09847) if you use Open3D.

```bib
@article{Zhou2018,
    author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
    title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
    journal   = {arXiv:1801.09847},
    year      = {2018},
}
```
