// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/t/geometry/TensorMap.h"

#include "open3d/t/geometry/PointCloud.h"
#include "pybind/docstring.h"
#include "pybind/t/geometry/geometry.h"

namespace open3d {
namespace t {
namespace geometry {

// This is a copy of py::bind_map function, with `__delitem__` function
// removed. The same is defined in `pybind_tensormap` to use
// `TensorMap::Erase(key)`.
template <typename Map,
          typename holder_type = std::unique_ptr<Map>,
          typename... Args>
static py::class_<Map, holder_type> bind_tensor_map(py::handle scope,
                                                    const std::string &name,
                                                    Args &&... args) {
    using KeyType = typename Map::key_type;
    using MappedType = typename Map::mapped_type;
    using Class_ = py::class_<Map, holder_type>;

    // If either type is a non-module-local bound type then make the map binding
    // non-local as well; otherwise (e.g. both types are either module-local or
    // converting) the map will be module-local.
    auto tinfo = py::detail::get_type_info(typeid(MappedType));
    bool local = !tinfo || tinfo->module_local;
    if (local) {
        tinfo = py::detail::get_type_info(typeid(KeyType));
        local = !tinfo || tinfo->module_local;
    }

    Class_ cl(scope, name.c_str(), pybind11::module_local(local),
              std::forward<Args>(args)...);

    cl.def(py::init<>());

    // Register stream insertion operator (if possible)
    py::detail::map_if_insertion_operator<Map, Class_>(cl, name);

    cl.def(
            "__bool__", [](const Map &m) -> bool { return !m.empty(); },
            "Check whether the map is nonempty");

    // Essential: keep list alive while iterator exists
    cl.def(
            "__iter__",
            [](Map &m) { return py::make_key_iterator(m.begin(), m.end()); },
            py::keep_alive<0, 1>()

    );

    // Essential: keep list alive while iterator exists
    cl.def(
            "items",
            [](Map &m) { return py::make_iterator(m.begin(), m.end()); },
            py::keep_alive<0, 1>());

    cl.def(
            "__getitem__",
            [](Map &m, const KeyType &k) -> MappedType & {
                auto it = m.find(k);
                if (it == m.end())
                    throw py::key_error(
                            fmt::format("Key {} not found in TensorMap", k));
                return it->second;
            },
            // py::return_value_policy::copy is used as the safest option.
            // The goal is to make TensorMap works similarly as putting Tensors
            // into a python dict, i.e., {"a": Tensor(xx), "b": Tensor(XX)}.
            // Accessing a value in the map will return a shallow copy of the
            // tensor that shares the same underlying memory.
            //
            // - automatic          : works, different id
            // - automatic_reference: works, different id
            // - take_ownership     : doesn't work, segfault
            // - copy               : works, different id
            // - move               : doesn't work, blob is null
            // - reference          : doesn't work, when a key is deleted, the
            //                        alias becomes invalid
            // - reference_internal : doesn't work, value in map overwritten
            //                        when assigning to alias
            py::return_value_policy::copy);

    cl.def("__setitem__", [](Map &m, const KeyType &k, const MappedType &v) {
        if (!TensorMap::GetReservedKeys().count(k)) {
            m[k] = v;
        } else {
            throw py::key_error(
                    fmt::format("Cannot assign to reserved key \"{}\"", k));
        }
    });

    cl.def("__contains__", [](Map &m, const KeyType &k) -> bool {
        auto it = m.find(k);
        if (it == m.end()) return false;
        return true;
    });

    // Assignment provided only if the type is copyable
    py::detail::map_assignment<Map, Class_>(cl);

    // Deleted the "__delitem__" function.
    // This will be implemented in `pybind_tensormap()`.

    cl.def("__len__", [](const Map &m) -> size_t { return m.size(); });

    return cl;
}

void pybind_tensormap_declarations(py::module &m) {
    auto tm = bind_tensor_map<TensorMap>(
            m, "TensorMap", "Map of String to Tensor with a primary key.");
}
void pybind_tensormap_definitions(py::module &m) {
    // Bind to the generic dictionary interface such that it works the same as a
    // regular dictionary in Python, except that types are enforced. Supported
    // functions include `__bool__`, `__iter__`, `items`, `__getitem__`,
    // `__contains__`, `__len__` and map assignment.
    // The `__delitem__` function is removed from bind_map, in bind_tensor_map,
    // and defined in this function, to use TensorMap::Erase, in order to
    // protect users from deleting the `private_key`.
    auto tm = static_cast<py::class_<TensorMap, std::unique_ptr<TensorMap>>>(
            m.attr("TensorMap"));
    tm.def("__delitem__",
           [](TensorMap &m, const std::string &k) { return m.Erase(k); });

    tm.def("erase",
           [](TensorMap &m, const std::string &k) { return m.Erase(k); });

    // Constructors.
    tm.def(py::init<const std::string &>(), "primary_key"_a);
    tm.def(py::init<const std::string &,
                    const std::unordered_map<std::string, core::Tensor> &>(),
           "primary_key"_a, "map_keys_to_tensors"_a);

    // Member functions. Some C++ functions are ignored since the
    // functionalities are already covered in the generic dictionary interface.
    tm.def_property_readonly("primary_key", &TensorMap::GetPrimaryKey);
    tm.def("is_size_synchronized", &TensorMap::IsSizeSynchronized);
    tm.def("assert_size_synchronized", &TensorMap::AssertSizeSynchronized);

    // Pickle support.
    tm.def(py::pickle(
            [](const TensorMap &m) {
                // __getstate__
                std::unordered_map<std::string, core::Tensor> map;
                for (const auto &kv : m) {
                    map[kv.first] = kv.second;
                }

                return py::make_tuple(m.GetPrimaryKey(), map);
            },
            [](py::tuple t) {
                // __setstate__
                if (t.size() != 2) {
                    utility::LogError(
                            "Cannot unpickle TensorMap! Expecting a tuple of "
                            "size 2.");
                }
                return TensorMap(t[0].cast<std::string>(),
                                 t[1].cast<std::unordered_map<std::string,
                                                              core::Tensor>>());
            }));

    tm.def("__setattr__",
           [](TensorMap &m, const std::string &key, const core::Tensor &val) {
               if (!TensorMap::GetReservedKeys().count(key)) {
                   m[key] = val;
               } else {
                   throw py::key_error(fmt::format(
                           "Cannot assign to reserved key \"{}\"", key));
               }
           });

    tm.def("__getattr__",
           [](TensorMap &m, const std::string &key) -> core::Tensor {
               auto it = m.find(key);
               if (it == m.end()) {
                   throw py::key_error(
                           fmt::format("Key {} not found in TensorMap", key));
               }
               return it->second;
           });

    tm.def("__delattr__", [](TensorMap &m, const std::string &key) {
        auto it = m.find(key);
        if (it == m.end()) {
            throw py::key_error(
                    fmt::format("Key {} not found in TensorMap", key));
        }
        return m.Erase(key);
    });

    tm.def("__str__", &TensorMap::ToString);

    tm.def("__repr__", &TensorMap::ToString);

    tm.def("__dir__", [](TensorMap &m) {
        auto keys = py::list();
        for (const auto &kv : m) {
            keys.append(kv.first);
        }
        return keys;
    });
}

}  // namespace geometry
}  // namespace t
}  // namespace open3d
