<p align="center">
<img src="docs/_static/open3d_logo_horizontal.png" width="320" />
</p>

# Open3D: A Modern Library for 3D Data Processing

**[Homepage](http://www.open3d.org") |**
**[Docs](http://www.open3d.org/docs") |**
**[Viewer App](https://github.com/intel-isl/Open3D/releases") |**
**[Quick Start](http://www.open3d.org/docs/release/getting_started.html") |**
**[Build from Source](http://www.open3d.org/docs/release/compilation.html") |**
**[Python API](http://www.open3d.org/docs/release/index.html#python-api-index") |**
**[C++ API](http://www.open3d.org/docs/release/cpp_api/index.html") |**
**[Contribute](http://www.open3d.org/docs/release/contribute.html") |**
**[Demo](https://www.youtube.com/watch?v=I3UjXlA4IsU") |**
**[Forum](https://forum.open3d.org")**

Open3D is an open-source library that supports rapid development of software
that deals with 3D data. The Open3D frontend exposes a set of carefully selected
data structures and algorithms in both C++ and Python. The backend is highly
optimized and is set up for parallelization. We welcome contributions from
the open-source community.

[![C/C++ CI](https://github.com/intel-isl/Open3D/workflows/C/C++%20CI/badge.svg)](https://github.com/intel-isl/Open3D/actions)
[![Build Status](https://travis-ci.org/intel-isl/Open3D.svg?branch=master)](https://travis-ci.org/intel-isl/)
[![Build status](https://ci.appveyor.com/api/projects/status/3hasjo041lv6srsi/branch/master?svg=true)](https://ci.appveyor.com/project/yxlao/open3d/branch/master)

Core features of Open3D includes:

* 3D data structures
* 3D data processing algorithms
* Scene reconstruction
* Surface alignment
* 3D visualization
* Available in C++ and Python

For more, please visit the [Open3D documentation](http://www.open3d.org/docs).

## Python quick start

Pre-built pip and conda packages support Ubuntu 18.04+, up-to-date macOS and
Windows 64-bit with Python 3.5, 3.6, 3.7 and 3.8. If you have other Python
versions or operating systems, please
[compile Open3D from source](http://www.open3d.org/docs/release/compilation.html).

* To install Open3D with pip:

    ```bash
    $ pip install open3d
    ```

* To install Open3D with Conda:

    ```bash
    $ conda install -c open3d-admin open3d
    ```

* To compile Open3D from source:
    * See [compiling from source](http://www.open3d.org/docs/release/compilation.html).

Test your installation with:

```bash
$ python -c "import open3d as o3d"
```

and follow the [basic tutorials](http://www.open3d.org/docs/release/tutorial/Basic/index.html)
or [Python examples](https://github.com/intel-isl/Open3D/tree/master/examples/Python) to get
started.

## C++ quick start

Please refer to [compiling from source](http://www.open3d.org/docs/release/compilation.html)
and [Open3D C++ interface](http://www.open3d.org/docs/release/tutorial/C++/cplusplus_interface.html).

## Open3D standalone viewer app (New!)

Open3D now comes with a standalone
[3D viewer app](https://github.com/intel-isl/Open3D/releases) available on
Ubuntu and macOS. Stay tuned for Windows.

## Communication channels

* [GitHub Issue](https://github.com/intel-isl/Open3D/issues): bug reports,
  feature requests, etc.
* [Forum](https://forum.open3d.org): discussion on the usage of Open3D.
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
