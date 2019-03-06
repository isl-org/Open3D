// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <string>
#include <unordered_map>
#include <tuple>

#include "TestUtility/UnitTest.h"

// Splits
// "create_mesh_arrow(cylinder_radius: float = 1.0) -> geometry.TriangleMesh"
// to
// ("create_mesh_arrow(cylinder_radius: float = 1.0)",
//  "geometry.TriangleMesh")
// std::pair<std::string, std::string> split_arrow(const std::string& docs) {
//     std::size_t arrow_pos = docs.rfind(" -> ");
//     std::string func_name_and_param = docs.substr(0, arrow_pos);
// }

// Currently copied this function for testing
// TODO: link unit test with python module to enable direct testing
std::pair<std::unordered_map<std::string, std::string>,
          std::vector<std::string>>
parse_pybind_function_doc(const std::string& pybind_docs) {
    std::unordered_map<std::string, std::string> map_parameter_type_docs;
    std::vector<std::string> ordered_parameters;

    // Split by "->"

    return std::make_pair(map_parameter_type_docs, ordered_parameters);
}

TEST(parse_pybind_function_doc, test_docstring_parse) {
    std::string docs = R"(
create_mesh_arrow(cylinder_radius: float = 1.0, cone_split: int = 1) -> open3d.open3d.geometry.TriangleMesh

Factory function to create an arrow mesh
)";
    parse_pybind_function_doc(docs);
}
