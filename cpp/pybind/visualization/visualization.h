// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "pybind/open3d_pybind.h"
#include "pybind11/functional.h"

// We cannot give out a shared_ptr to objects like Window which reference
// Filament objects, because we cannot guarantee that the Python script is
// not holding on to a reference when we cleanup Filament. The Open3D library
// will clear its shared_ptrs expecting the dependent object(s) to clean up,
// but they won't because Python still has a shared_ptr, leading to a crash
// when the variable goes of scope on the Python side.
// The following would crash gui.Window's holder is std::shared_ptr:
//   import open3d.visualization.gui as gui
//   def main():
//       gui.Application.instance.initialize()
//       w = gui.Application.instance.create_window("Crash", 640, 480)
//       gui.Application.instance.run()
//   if __name__ == "__main__":
//       main()
// However, if remove the 'w = ' part, it would not crash.
template <typename T>
class UnownedPointer {
public:
    UnownedPointer() : ptr_(nullptr) {}
    explicit UnownedPointer(T *p) : ptr_(p) {}
    ~UnownedPointer() {}  // don't delete!

    T *get() { return ptr_; }
    T &operator*() { return *ptr_; }
    T *operator->() { return ptr_; }
    void reset() { ptr_ = nullptr; }  // don't delete!

private:
    T *ptr_;
};
PYBIND11_DECLARE_HOLDER_TYPE(T, UnownedPointer<T>);

namespace open3d {
namespace visualization {

template <typename T>
std::shared_ptr<T> TakeOwnership(UnownedPointer<T> x) {
    return std::shared_ptr<T>(x.get());
}

// Please update docs/python_api_in/open3d.visualization.rst when adding /
// reorganizing submodules to open3d.visualization.

void pybind_visualization(py::module &m);

void pybind_renderoption(py::module &m);
void pybind_viewcontrol(py::module &m);
void pybind_visualizer(py::module &m);
void pybind_visualization_utility(py::module &m);

void pybind_renderoption_method(py::module &m);
void pybind_viewcontrol_method(py::module &m);
void pybind_visualizer_method(py::module &m);
void pybind_visualization_utility_methods(py::module &m);

void pybind_o3dvisualizer(py::module &m);

}  // namespace visualization
}  // namespace open3d
