// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/utility/Console.h"

#include <Eigen/Core>
#include <string>
#include <vector>

#include "open3d/utility/Helper.h"

namespace open3d {
namespace utility {

std::string GetProgramOptionAsString(
        int argc,
        char **argv,
        const std::string &option,
        const std::string &default_value /* = ""*/) {
    char **itr = std::find(argv, argv + argc, option);
    if (itr != argv + argc && ++itr != argv + argc) {
        return std::string(*itr);
    }
    return default_value;
}

int GetProgramOptionAsInt(int argc,
                          char **argv,
                          const std::string &option,
                          const int default_value /* = 0*/) {
    std::string str = GetProgramOptionAsString(argc, argv, option, "");
    if (str.length() == 0) {
        return default_value;
    }
    char *end;
    errno = 0;
    long l = std::strtol(str.c_str(), &end, 0);
    if ((errno == ERANGE && l == LONG_MAX) || l > INT_MAX) {
        return default_value;
    } else if ((errno == ERANGE && l == LONG_MIN) || l < INT_MIN) {
        return default_value;
    } else if (*end != '\0') {
        return default_value;
    }
    return (int)l;
}

double GetProgramOptionAsDouble(int argc,
                                char **argv,
                                const std::string &option,
                                const double default_value /* = 0.0*/) {
    std::string str = GetProgramOptionAsString(argc, argv, option, "");
    if (str.length() == 0) {
        return default_value;
    }
    char *end;
    errno = 0;
    double l = std::strtod(str.c_str(), &end);
    if (errno == ERANGE && (l == HUGE_VAL || l == -HUGE_VAL)) {
        return default_value;
    } else if (*end != '\0') {
        return default_value;
    }
    return l;
}

Eigen::VectorXd GetProgramOptionAsEigenVectorXd(
        int argc,
        char **argv,
        const std::string &option,
        const Eigen::VectorXd default_value /* =
        Eigen::VectorXd::Zero()*/) {
    std::string str = GetProgramOptionAsString(argc, argv, option, "");
    if (str.length() == 0 || (!(str.front() == '(' && str.back() == ')') &&
                              !(str.front() == '[' && str.back() == ']') &&
                              !(str.front() == '<' && str.back() == '>'))) {
        return default_value;
    }
    std::vector<std::string> tokens =
            SplitString(str.substr(1, str.length() - 2), ",");
    Eigen::VectorXd vec(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        char *end;
        errno = 0;
        double l = std::strtod(tokens[i].c_str(), &end);
        if (errno == ERANGE && (l == HUGE_VAL || l == -HUGE_VAL)) {
            return default_value;
        } else if (*end != '\0') {
            return default_value;
        }
        vec(i) = l;
    }
    return vec;
}

bool ProgramOptionExists(int argc, char **argv, const std::string &option) {
    return std::find(argv, argv + argc, option) != argv + argc;
}

bool ProgramOptionExistsAny(int argc,
                            char **argv,
                            const std::vector<std::string> &options) {
    for (const auto &option : options) {
        if (ProgramOptionExists(argc, argv, option)) {
            return true;
        }
    }
    return false;
}

}  // namespace utility
}  // namespace open3d
