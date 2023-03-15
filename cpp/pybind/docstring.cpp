// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "pybind/docstring.h"

#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "open3d/utility/Helper.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace docstring {

// ref: enum_base in pybind11.h
py::handle static_property =
        py::handle((PyObject*)py::detail::get_internals().static_property_type);

void ClassMethodDocInject(py::module& pybind_module,
                          const std::string& class_name,
                          const std::string& function_name,
                          const std::unordered_map<std::string, std::string>&
                                  map_parameter_body_docs) {
    // Get function
    PyObject* module = pybind_module.ptr();
    PyObject* class_obj = PyObject_GetAttrString(module, class_name.c_str());
    if (class_obj == nullptr) {
        utility::LogWarning("{} docstring failed to inject.", class_name);
        return;
    }
    PyObject* class_method_obj =
            PyObject_GetAttrString(class_obj, function_name.c_str());
    if (class_method_obj == nullptr) {
        utility::LogWarning("{}::{} docstring failed to inject.", class_name,
                            function_name);
        return;
    }

    // Extract PyCFunctionObject
    PyCFunctionObject* f = nullptr;
    if (Py_TYPE(class_method_obj) == &PyInstanceMethod_Type) {
        PyInstanceMethodObject* class_method =
                (PyInstanceMethodObject*)class_method_obj;
        f = (PyCFunctionObject*)class_method->func;
    }
    if (Py_TYPE(class_method_obj) == &PyCFunction_Type) {
        // def_static in Pybind is PyCFunction_Type, no need to convert
        f = (PyCFunctionObject*)class_method_obj;
    }
    if (f == nullptr || Py_TYPE(f) != &PyCFunction_Type) {
        return;
    }

    // Parse existing docstring to FunctionDoc
    FunctionDoc fd(f->m_ml->ml_doc);

    // Inject docstring
    for (auto& overload : fd.overload_docs_) {
        for (ArgumentDoc& ad : overload.argument_docs_) {
            if (map_parameter_body_docs.count(ad.name_) > 0) {
                ad.body_ = map_parameter_body_docs.at(ad.name_);
            }
        }
    }
    f->m_ml->ml_doc = strdup(fd.ToGoogleDocString().c_str());
}

void FunctionDocInject(py::module& pybind_module,
                       const std::string& function_name,
                       const std::unordered_map<std::string, std::string>&
                               map_parameter_body_docs) {
    // Get function
    PyObject* module = pybind_module.ptr();
    PyObject* f_obj = PyObject_GetAttrString(module, function_name.c_str());
    if (f_obj == nullptr) {
        utility::LogWarning("{} docstring failed to inject.", function_name);
        return;
    }
    if (Py_TYPE(f_obj) != &PyCFunction_Type) {
        return;
    }
    PyCFunctionObject* f = (PyCFunctionObject*)f_obj;

    // Parse existing docstring to FunctionDoc
    FunctionDoc fd(f->m_ml->ml_doc);

    // Inject docstring: repeat for each overload
    for (auto& overload : fd.overload_docs_) {
        for (ArgumentDoc& ad : overload.argument_docs_) {
            if (map_parameter_body_docs.count(ad.name_) > 0) {
                ad.body_ = map_parameter_body_docs.at(ad.name_);
            }
        }
    }
    f->m_ml->ml_doc = strdup(fd.ToGoogleDocString().c_str());
}

FunctionDoc::FunctionDoc(const std::string& pybind_doc)
    : pybind_doc_(pybind_doc) {
    doc_pos_[0] = ParseFunctionName();
    doc_pos_[1] = ParseSummary();
    // Repeat for each overload:
    for (; doc_pos_[1] != std::string::npos; doc_pos_[1] = ParseSummary()) {
        ParseArguments();
        ParseReturn();
        doc_pos_[0] = doc_pos_[1];
    }
}

size_t FunctionDoc::ParseFunctionName() {
    size_t parenthesis_pos = pybind_doc_.find("(");
    if (parenthesis_pos != std::string::npos) {
        std::string name = pybind_doc_.substr(0, parenthesis_pos);
        name_ = name;
    }
    size_t preamble_end = pybind_doc_.find("Overloaded function.");
    if (preamble_end == std::string::npos) {
        return parenthesis_pos;
    } else {
        preamble_end += strlen("Overloaded function.");
        preamble_ = pybind_doc_.substr(0, preamble_end);
        return preamble_end;
    }
}

size_t FunctionDoc::ParseSummary() {
    size_t arrow_pos = pybind_doc_.find(" -> ", doc_pos_[0]);
    size_t summary_end_pos = std::string::npos;
    if (arrow_pos != std::string::npos) {
        overload_docs_.push_back(OverloadDocs{});
        size_t result_type_pos = arrow_pos + 4;
        size_t summary_start_pos =
                result_type_pos +
                utility::WordLength(pybind_doc_, result_type_pos, "._:,[]() ,");
        summary_end_pos =
                pybind_doc_.find(". " + name_ + "(", summary_start_pos);
        if (summary_end_pos == std::string::npos)
            summary_end_pos = pybind_doc_.size();  // Last overload
        else
            summary_end_pos -= 3;  // \n\n[:digit:]
        size_t summary_len = summary_end_pos - summary_start_pos;
        if (summary_len > 0) {
            std::string summary =
                    pybind_doc_.substr(summary_start_pos, summary_len);
            overload_docs_.back().summary_ = StringCleanAll(summary);
        }
    }
    return summary_end_pos;
}

void FunctionDoc::ParseArguments() {
    // Parse docstrings of arguments
    // Input: "foo(arg0: float, arg1: float = 1.0, arg2: int = 1) -> open3d.bar"
    // Goal: split to {"arg0: float", "arg1: float = 1.0", "arg2: int = 1"} and
    //       call function to parse each argument respectively
    std::vector<std::string> argument_tokens = GetArgumentTokens(
            pybind_doc_.substr(doc_pos_[0], doc_pos_[1] - doc_pos_[0]));
    overload_docs_.back().argument_docs_.clear();
    for (const std::string& argument_token : argument_tokens) {
        overload_docs_.back().argument_docs_.push_back(
                ParseArgumentToken(argument_token));
    }
}

void FunctionDoc::ParseReturn() {
    size_t arrow_pos = pybind_doc_.rfind(" -> ", doc_pos_[1]);
    if (arrow_pos != std::string::npos && arrow_pos > doc_pos_[0]) {
        size_t result_type_pos = arrow_pos + 4;
        std::string return_type = pybind_doc_.substr(
                result_type_pos,
                utility::WordLength(pybind_doc_, result_type_pos,
                                    "._:,[]() ,"));
        overload_docs_.back().return_doc_.type_ = StringCleanAll(return_type);
    }
}

std::string FunctionDoc::ToGoogleDocString() const {
    // Example Google style:
    // http://www.sphinx-doc.org/en/1.5/ext/example_google.html

    std::ostringstream rc;
    std::string indent = "    ";
    size_t n_overload = 1;

    if (!preamble_.empty()) {
        rc << preamble_ << std::endl << std::endl;
    }

    for (auto& overload : overload_docs_) {
        // Function signature to be parsed by Sphinx
        if (!preamble_.empty()) rc << std::endl << n_overload++ << ". ";
        rc << name_ << "(";
        for (size_t i = 0; i < overload.argument_docs_.size(); ++i) {
            const ArgumentDoc& argument_doc = overload.argument_docs_[i];
            rc << argument_doc.name_;
            if (argument_doc.default_ != "") {
                rc << "=" << argument_doc.default_;
            }
            if (i != overload.argument_docs_.size() - 1) {
                rc << ", ";
            }
        }
        rc << ")" << std::endl;

        // Summary line, strictly speaking this shall be at the very front.
        // However from a compiled Python module we need the function signature
        // hints in front for Sphinx parsing and PyCharm autocomplete
        if (overload.summary_ != "") {
            if (!preamble_.empty()) rc << indent;
            rc << overload.summary_ << std::endl;
        }

        // Arguments
        if (overload.argument_docs_.size() != 0 &&
            !(overload.argument_docs_.size() == 1 &&
              overload.argument_docs_[0].name_ == "self")) {
            rc << std::endl;
            rc << "Args:" << std::endl;
            for (const ArgumentDoc& argument_doc : overload.argument_docs_) {
                if (argument_doc.name_ == "self") {
                    continue;
                }
                rc << indent << argument_doc.name_ << " ("
                   << argument_doc.type_;
                if (argument_doc.default_ != "") {
                    rc << ", optional";
                }
                if (argument_doc.default_ != "" &&
                    argument_doc.long_default_ == "") {
                    rc << ", default=" << argument_doc.default_;
                }
                rc << ")";
                if (argument_doc.body_ != "") {
                    rc << ": " << argument_doc.body_;
                }
                if (argument_doc.long_default_ != "") {
                    std::vector<std::string> lines = utility::SplitString(
                            argument_doc.long_default_, "\n", true);
                    rc << " Default value:" << std::endl << std::endl;
                    bool prev_line_is_listing = false;
                    for (std::string& line : lines) {
                        line = StringCleanAll(line);
                        if (line[0] == '-') {  // listing
                            // Add empty line before listing
                            if (!prev_line_is_listing) {
                                rc << std::endl;
                            }
                            prev_line_is_listing = true;
                        } else {
                            prev_line_is_listing = false;
                        }
                        rc << indent << indent << line << std::endl;
                    }
                } else {
                    rc << std::endl;
                }
            }
        }

        // Return
        if (name_ != "__init__") {
            rc << std::endl;
            rc << "Returns:" << std::endl;
            rc << indent << overload.return_doc_.type_;
            if (overload.return_doc_.body_ != "") {
                rc << ": " << overload.return_doc_.body_;
            }
            rc << std::endl;
        }
    }
    return rc.str();
}

std::string FunctionDoc::NamespaceFix(const std::string& s) {
    std::string rc = std::regex_replace(s, std::regex("::(\\S)"), ".$1");
    rc = std::regex_replace(rc, std::regex("open3d\\.(cpu|cuda)\\.pybind\\."),
                            "open3d.");
    return rc;
}

std::string FunctionDoc::StringCleanAll(std::string& s,
                                        const std::string& white_space) {
    std::string rc = utility::StripString(s, white_space);
    rc = NamespaceFix(rc);
    return rc;
}

ArgumentDoc FunctionDoc::ParseArgumentToken(const std::string& argument_token) {
    ArgumentDoc argument_doc;

    // Argument with default value
    std::regex rgx_with_default(
            "([A-Za-z_][A-Za-z\\d_]*): "
            "([A-Za-z_][A-Za-z\\d_:\\.\\[\\]\\(\\) ,]*) = (.*)");
    std::smatch matches;
    if (std::regex_search(argument_token, matches, rgx_with_default)) {
        argument_doc.name_ = matches[1].str();
        argument_doc.type_ = NamespaceFix(matches[2].str());
        argument_doc.default_ = matches[3].str();

        // Handle long default value. Long default has multiple lines and thus
        // they are not displayed in  signature, but in docstrings.
        size_t default_start_pos = matches.position(3);
        if (default_start_pos + argument_doc.default_.size() <
            argument_token.size()) {
            argument_doc.long_default_ = argument_token.substr(
                    default_start_pos,
                    argument_token.size() - default_start_pos);
            argument_doc.default_ = "(with default value)";
        }
    }

    else {
        // Argument without default value
        std::regex rgx_without_default(
                "([A-Za-z_][A-Za-z\\d_]*): "
                "([A-Za-z_][A-Za-z\\d_:\\.\\[\\]\\(\\) ,]*)");
        if (std::regex_search(argument_token, matches, rgx_without_default)) {
            argument_doc.name_ = matches[1].str();
            argument_doc.type_ = NamespaceFix(matches[2].str());
        }
    }

    return argument_doc;
}

std::vector<std::string> FunctionDoc::GetArgumentTokens(
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
    for (size_t i = 0; i + 1 < argument_start_positions.size(); ++i) {
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

}  // namespace docstring
}  // namespace open3d
