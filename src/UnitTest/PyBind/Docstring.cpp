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
#include <unordered_set>

#include "TestUtility/UnitTest.h"

class ArgumentDoc {
public:
    std::string argument_name_;
    std::string argument_type_;
    std::string argument_default_;
};

class FunctionDoc {
public:
    std::string function_name_;
    std::vector<ArgumentDoc> argument_docs_;
    std::string summary_;
    std::string body_;
};

// Search and replace in string
std::string str_replace(std::string s,
                        const std::string& search,
                        const std::string& replace) {
    // https://stackoverflow.com/a/14679003/1255535
    size_t pos = 0;
    while ((pos = s.find(search, pos)) != std::string::npos) {
        s.replace(pos, search.length(), replace);
        pos += replace.length();
    }
    return s;
}

// Deduplicate namespace (optional)
std::string namespace_dedup(const std::string& s) {
    return str_replace(s, "open3d.open3d", "open3d");
}

// Similar to Python's str.strip()
std::string str_strip(const std::string& s,
                      const std::string& white_space = " \t\n") {
    size_t begin_pos = s.find_first_not_of(white_space);
    if (begin_pos == std::string::npos) {
        return "";
    }
    size_t end_pos = s.find_last_not_of(white_space);
    return s.substr(begin_pos, end_pos - begin_pos + 1);
}

// Count the length of current word starting from start_pos
size_t word_length(const std::string& doc,
                   size_t start_pos,
                   const std::string& valid_chars = "_") {
    std::unordered_set<char> valid_chars_set;
    for (const char& c : valid_chars) {
        valid_chars_set.insert(c);
    }
    auto is_word_char = [&valid_chars_set](const char& c) {
        return std::isalnum(c) ||
               valid_chars_set.find(c) != valid_chars_set.end();
    };
    size_t length = 0;
    for (size_t pos = start_pos; pos < doc.size(); ++pos) {
        if (!is_word_char(doc[pos])) {
            break;
        }
        length++;
    }
    return length;
}

// Splits
// "create_mesh_arrow(cylinder_radius: float = 1.0) -> geometry.TriangleMesh"
// to
// ("create_mesh_arrow(cylinder_radius: float = 1.0)",
//  "geometry.TriangleMesh")
std::pair<std::string, std::string> split_arrow(const std::string& doc) {
    std::size_t arrow_pos = doc.rfind(" -> ");
    if (arrow_pos != std::string::npos) {
        std::string func_name_and_params = doc.substr(0, arrow_pos);
        std::string return_type = doc.substr(
                arrow_pos + 4, word_length(doc, arrow_pos + 4, "._"));
        return std::make_pair(namespace_dedup(str_strip(func_name_and_params)),
                              namespace_dedup(str_strip(return_type)));
    } else {
        return std::make_pair(doc, "");
    }
}

// Currently copied this function for testing
// TODO: link unit test with python module to enable direct testing
std::pair<std::unordered_map<std::string, std::string>,
          std::vector<std::string>>
parse_pybind_function_doc(const std::string& pybind_doc) {
    std::unordered_map<std::string, std::string> map_parameter_type_docs;
    std::vector<std::string> ordered_parameters;

    // Split by "->"
    std::string func_name_and_params;
    std::string return_type;
    std::tie(func_name_and_params, return_type) = split_arrow(pybind_doc);

    std::cout << "func_name_and_params " << func_name_and_params << std::endl;
    std::cout << "return_type " << return_type << std::endl;

    return std::make_pair(map_parameter_type_docs, ordered_parameters);
}

TEST(parse_pybind_function_doc, test_docstring_parse) {
    std::string doc = R"(
create_mesh_arrow(cylinder_radius: float = 1.0, cone_split: int = 1) -> open3d.open3d.geometry.TriangleMesh

Factory function to create an arrow mesh
)";
    parse_pybind_function_doc(doc);
}
