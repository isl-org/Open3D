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
#include <sstream>
#include <regex>

#include "Python/docstring.h"

namespace open3d {
namespace docstring {

// Deduplicate namespace (optional)
static std::string namespace_fix(const std::string& s) {
    std::string rc = std::regex_replace(s, std::regex("::"), ".");
    rc = std::regex_replace(rc, std::regex("open3d\\.open3d\\."), "open3d.");
    return rc;
}

// Similar to Python's str.strip()
static std::string str_strip(const std::string& s,
                             const std::string& white_space = " \t\n") {
    size_t begin_pos = s.find_first_not_of(white_space);
    if (begin_pos == std::string::npos) {
        return "";
    }
    size_t end_pos = s.find_last_not_of(white_space);
    return s.substr(begin_pos, end_pos - begin_pos + 1);
}

// Run all string cleanup functions
static std::string str_clean_all(const std::string& s,
                                 const std::string& white_space = " \t\n") {
    std::string rc = str_strip(s, white_space);
    rc = namespace_fix(rc);
    return rc;
}

// Count the length of current word starting from start_pos
static size_t word_length(const std::string& doc,
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

static void parse_function_name(const std::string& pybind_doc,
                                FunctionDoc& function_doc) {
    size_t parenthesis_pos = pybind_doc.find("(");
    if (parenthesis_pos == std::string::npos) {
        return;
    } else {
        std::string name = pybind_doc.substr(0, parenthesis_pos);
        function_doc.name_ = name;
    }
}

// Parse docstring for a single argument
// E.g. "cylinder_radius: float = 1.0"
// E.g. "cylinder_radius: float"
static ArgumentDoc parse_argument_token(const std::string& argument_token) {
    // std::cout << "argument_token:" << std::endl << argument_token <<
    // std::endl;
    ArgumentDoc argument_doc;

    // Argument with default value
    std::regex rgx_with_default(
            "([A-Za-z_][A-Za-z\\d_]*): ([A-Za-z_][A-Za-z\\d_:\\.]*) = (.*)");
    std::smatch matches;
    if (std::regex_search(argument_token, matches, rgx_with_default)) {
        argument_doc.name_ = matches[1].str();
        argument_doc.type_ = namespace_fix(matches[2].str());
        argument_doc.default_ = matches[3].str();
    } else {
        // Argument without default value
        std::regex rgx_without_default(
                "([A-Za-z_][A-Za-z\\d_]*): ([A-Za-z_][A-Za-z\\d_:\\.]*)");
        if (std::regex_search(argument_token, matches, rgx_without_default)) {
            argument_doc.name_ = matches[1].str();
            argument_doc.type_ = namespace_fix(matches[2].str());
        }
    }

    return argument_doc;
}

static std::vector<std::string> get_argument_tokens(
        const std::string& pybind_doc) {
    // First insert commas to make things easy
    // From:
    // "foo(arg0: float, arg1: float = 1.0, arg2: int = 1) -> open3d.bar"
    // To:
    // "foo(, arg0: float, arg1: float = 1.0, arg2: int = 1) -> open3d.bar"
    std::string str = pybind_doc;
    size_t parenthesis_pos = str.find("(");
    if (parenthesis_pos == std::string::npos) {
        return {};
    } else {
        str.replace(parenthesis_pos + 1, 0, ", ");
    }

    // Get start positions
    std::regex pattern("(, [A-Za-z_][A-Za-z\\d_]*:)");
    std::smatch res;
    std::string::const_iterator start_iter(str.cbegin());
    std::vector<size_t> argument_start_positions;
    while (std::regex_search(start_iter, str.cend(), res, pattern)) {
        size_t pos = res.position(0) + (start_iter - str.cbegin());
        start_iter = res.suffix().first;
        // Now the pos include ", ", which needs to be removed
        argument_start_positions.push_back(pos + 2);
    }

    // Get end positions (non-inclusive)
    // The 1st argument's end pos is 2nd argument's start pos - 2 and etc.
    // The last argument's end pos is the location of the parenthesis before ->
    std::vector<size_t> argument_end_positions;
    for (size_t i = 0; i < argument_start_positions.size() - 1; ++i) {
        argument_end_positions.push_back(argument_start_positions[i + 1] - 2);
    }
    std::size_t arrow_pos = str.rfind(") -> ");
    if (arrow_pos == std::string::npos) {
        return {};
    } else {
        argument_end_positions.push_back(arrow_pos);
    }

    std::vector<std::string> argument_tokens;
    for (size_t i = 0; i < argument_start_positions.size(); ++i) {
        std::string token = str.substr(
                argument_start_positions[i],
                argument_end_positions[i] - argument_start_positions[i]);
        argument_tokens.push_back(token);
    }
    return argument_tokens;
}

// Parse docstrings of arguments
// Input: "foo(arg0: float, arg1: float = 1.0, arg2: int = 1) -> open3d.bar"
// Goal: split to {"arg0: float", "arg1: float = 1.0", "arg2: int = 1"} and call
//       parse_single_argument respectively
static void parse_doc_arguments(const std::string& pybind_doc,
                                FunctionDoc& function_doc) {
    std::vector<std::string> argument_tokens = get_argument_tokens(pybind_doc);
    function_doc.argument_docs_.clear();
    for (const std::string& argument_token : argument_tokens) {
        function_doc.argument_docs_.push_back(
                parse_argument_token(argument_token));
    }
}

static void parse_doc_result(const std::string& pybind_doc,
                             FunctionDoc& function_doc) {
    size_t arrow_pos = pybind_doc.rfind(" -> ");
    if (arrow_pos != std::string::npos) {
        size_t result_type_pos = arrow_pos + 4;
        std::string return_type = pybind_doc.substr(
                result_type_pos,
                word_length(pybind_doc, result_type_pos, "._:"));
        function_doc.return_doc_.type_ = str_clean_all(return_type);
    }
}

// Currently copied this function for testing
// TODO: link unit test with python module to enable direct testing
static FunctionDoc parse_doc_function(const std::string& pybind_doc) {
    FunctionDoc function_doc(pybind_doc);
    parse_function_name(pybind_doc, function_doc);
    parse_doc_arguments(pybind_doc, function_doc);
    parse_doc_result(pybind_doc, function_doc);
    return function_doc;
}

void function_doc_inject(py::module& pybind_module,
                         const std::string& function_name,
                         const std::unordered_map<std::string, std::string>
                                 map_parameter_body_docs) {
    // Get function
    PyObject* module = pybind_module.ptr();
    PyObject* f_obj = PyObject_GetAttrString(module, function_name.c_str());
    if (Py_TYPE(f_obj) != &PyCFunction_Type) {
        return;
    }
    PyCFunctionObject* f = (PyCFunctionObject*)f_obj;

    FunctionDoc fd = parse_doc_function(f->m_ml->ml_doc);
    for (const auto& it : map_parameter_body_docs) {
        fd.inject_argument_doc_body(it.first, it.second);
    }
    f->m_ml->ml_doc = strdup(fd.to_string().c_str());
}

void FunctionDoc::inject_argument_doc_body(
        const std::string& argument_name,
        const std::string& argument_doc_body) {
    for (ArgumentDoc& argument_doc : argument_docs_) {
        if (argument_doc.name_ == argument_name) {
            argument_doc.body_ = argument_doc_body;
            break;
        }
    }
}

FunctionDoc::FunctionDoc(const std::string& pybind_doc)
    : pybind_doc_(pybind_doc) {
    parse_summary();
    parse_arguments();
    parse_result();
}

void FunctionDoc::parse_summary() {
    size_t arrow_pos = pybind_doc_.rfind(" -> ");
    if (arrow_pos != std::string::npos) {
        size_t result_type_pos = arrow_pos + 4;
        size_t summary_start_pos =
                result_type_pos +
                word_length(pybind_doc_, result_type_pos, "._:");
        size_t summary_len = pybind_doc_.size() - summary_start_pos;
        if (summary_len > 0) {
            std::string summary =
                    pybind_doc_.substr(summary_start_pos, summary_len);
            summary_ = str_clean_all(summary);
        }
    }
}

void FunctionDoc::parse_arguments() {}

void FunctionDoc::parse_result() {}

std::string FunctionDoc::to_string() const {
    // Example Gooele style:
    // http://www.sphinx-doc.org/en/1.5/ext/example_google.html

    std::ostringstream rc;
    std::string indent = "    ";

    // Function signature to be parsed by Sphinx
    rc << name_ << "(";
    for (size_t i = 0; i < argument_docs_.size(); ++i) {
        const ArgumentDoc& argument_doc = argument_docs_[i];
        rc << argument_doc.name_;
        if (argument_doc.default_ != "") {
            rc << "=" << argument_doc.default_;
        }
        if (i != argument_docs_.size() - 1) {
            rc << ", ";
        }
    }
    rc << ")" << std::endl;

    // Summary line, strictly speaking this shall be at the very front. However
    // from a compiled Python module we need the function signature hints in
    // front for Sphinx parsing and PyCharm autocomplete
    if (summary_ != "") {
        rc << std::endl;
        rc << summary_ << std::endl;
    }

    // Arguments
    if (argument_docs_.size() != 0) {
        rc << std::endl;
        rc << "Args:" << std::endl;
        for (const ArgumentDoc& argument_doc : argument_docs_) {
            rc << indent << argument_doc.name_ << " (" << argument_doc.type_;
            if (argument_doc.default_ != "") {
                rc << ", optional";
            }
            rc << ")";
            if (argument_doc.body_ != "") {
                rc << ": " << argument_doc.body_;
            }
            rc << std::endl;
        }
    }

    // Return
    rc << std::endl;
    rc << "Returns:" << std::endl;
    rc << indent << return_doc_.type_;
    if (return_doc_.body_ != "") {
        rc << ": " << return_doc_.body_;
    }
    rc << std::endl;

    return rc.str();
}

}  // namespace docstring
}  // namespace open3d
