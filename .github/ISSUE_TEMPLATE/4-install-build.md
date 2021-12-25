---
name: Installation and Build
about: I have trouble installing or compiling Open3D.
title: "Summarize the issue and your environments (e.g., \"Cannot compile on Ubuntu 20.04 with OpenBLAS\"")"
labels: [build/install issue]
---

Before submitting, I have (mark "x" in the check-box):
- [ ] Search for [similar issues](https://github.com/isl-org/Open3D/issues).
- [ ] Tested the [latest development wheel](http://www.open3d.org/docs/latest/getting_started.html#development-version-pip).

## To Reproduce
<!--
Please provide step-by-step instructions on how to reproduce the issue.
Providing *detailed* instructions is crucial for other to help with the issues.
Please modify the following example.
-->

I first clone Open3D by:

```bash
git clone https://github.com/isl-org/Open3D.git
cd Open3D
```

Then, I build Open3D with:

```bash
mkdir build
cd build
cmake -DUSE_BLAS=ON ..
make -j$(nproc)
```

## Error info and terminal outputs
<!--
Please:
- Provide the *full* error message.
- It is even better to attach your terminal commands and the full terminal outputs.
- Provide screenshots if applicable.
-->

When I build Open3D, I get the following error:

```
-- Building OpenBLAS with LAPACK from source
CMake Error at 3rdparty/find_dependencies.cmake:1227 (message):
  gfortran is required to compile LAPACK from source.  On Ubuntu, please
  install by `apt install gfortran`.  On macOS, please install by `brew
  install gfortran`.
Call Stack (most recent call first):
  CMakeLists.txt:446 (include)
```

Here's the full terminal output with my commands:
(drag and drop a .txt file to GitHub)


## Environment
<!--
Please specify your system environments, for example:
- Operating system: Ubuntu 18.04 (ARM64)
- Python version: Python 3.6
- Open3D version: Open3D master branch, commit xxxxx
- Is this remote workstation: No.
- How did you install Open3D: Compile from source.
- Compiler version (if build from source): gcc-9
-->
- Operating system:
- Python version:
- Open3D version:
- Is this remote workstation?:
- How did you install Open3D?:
- Compiler version (if build from source):
