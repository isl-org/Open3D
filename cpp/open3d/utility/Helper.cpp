// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

#include "open3d/utility/Helper.h"

#include <fmt/chrono.h>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_set>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif  // _WIN32

namespace open3d {
namespace utility {

std::vector<std::string> SplitString(const std::string& str,
                                     const std::string& delimiters /* = " "*/,
                                     bool trim_empty_str /* = true*/) {
    std::vector<std::string> tokens;
    std::string::size_type pos = 0, new_pos = 0, last_pos = 0;
    while (pos != std::string::npos) {
        pos = str.find_first_of(delimiters, last_pos);
        new_pos = (pos == std::string::npos ? str.length() : pos);
        if (new_pos != last_pos || !trim_empty_str) {
            tokens.push_back(str.substr(last_pos, new_pos - last_pos));
        }
        last_pos = new_pos + 1;
    }
    return tokens;
}

bool ContainsString(const std::string& src, const std::string& dst) {
    return src.find(dst) != std::string::npos;
}

bool StringStartsWith(const std::string& src, const std::string& tar) {
    // https://stackoverflow.com/a/42844629/1255535
    return src.size() >= tar.size() && 0 == src.compare(0, tar.size(), tar);
}

bool StringEndsWith(const std::string& src, const std::string& tar) {
    // https://stackoverflow.com/a/42844629/1255535
    return src.size() >= tar.size() &&
           0 == src.compare(src.size() - tar.size(), tar.size(), tar);
}

std::string JoinStrings(const std::vector<std::string>& strs,
                        const std::string& delimiter) {
    std::ostringstream oss;
    for (size_t i = 0; i < strs.size(); ++i) {
        oss << strs[i];
        if (i != strs.size() - 1) {
            oss << delimiter;
        }
    }
    return oss.str();
}

std::string& LeftStripString(std::string& str, const std::string& chars) {
    str.erase(0, str.find_first_not_of(chars));
    return str;
}

std::string& RightStripString(std::string& str, const std::string& chars) {
    str.erase(str.find_last_not_of(chars) + 1);
    return str;
}

std::string& StripString(std::string& str, const std::string& chars) {
    return LeftStripString(RightStripString(str, chars), chars);
}

std::string ToLower(const std::string& str) {
    std::string out = str;
    std::transform(str.begin(), str.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

std::string ToUpper(const std::string& str) {
    std::string out = str;
    std::transform(str.begin(), str.end(), out.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return out;
}

// Count the length of current word starting from start_pos
size_t WordLength(const std::string& doc,
                  size_t start_pos,
                  const std::string& valid_chars) {
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

void Sleep(int milliseconds) {
#ifdef _WIN32
    ::Sleep(milliseconds);
#else
    usleep(milliseconds * 1000);
#endif  // _WIN32
}

std::string GetCurrentTimeStamp() {
    std::time_t t = std::time(nullptr);
    return fmt::format("{:%Y-%m-%d-%H-%M-%S}", *std::localtime(&t));
}

}  // namespace utility
}  // namespace open3d
