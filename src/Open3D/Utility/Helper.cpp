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

#include "Open3D/Utility/Helper.h"

#include <cctype>
#include <unordered_set>

namespace open3d {
namespace utility {

void SplitString(std::vector<std::string> &tokens,
                 const std::string &str,
                 const std::string &delimiters /* = " "*/,
                 bool trim_empty_str /* = true*/) {
    std::string::size_type pos = 0, new_pos = 0, last_pos = 0;
    while (pos != std::string::npos) {
        pos = str.find_first_of(delimiters, last_pos);
        new_pos = (pos == std::string::npos ? str.length() : pos);
        if (new_pos != last_pos || !trim_empty_str) {
            tokens.push_back(str.substr(last_pos, new_pos - last_pos));
        }
        last_pos = new_pos + 1;
    }
}

std::string StripString(const std::string &s, const std::string &white_space) {
    size_t begin_pos = s.find_first_not_of(white_space);
    if (begin_pos == std::string::npos) {
        return "";
    }
    size_t end_pos = s.find_last_not_of(white_space);
    return s.substr(begin_pos, end_pos - begin_pos + 1);
}

// Count the length of current word starting from start_pos
size_t WordLength(const std::string &doc,
                  size_t start_pos,
                  const std::string &valid_chars) {
    std::unordered_set<char> valid_chars_set;
    for (const char &c : valid_chars) {
        valid_chars_set.insert(c);
    }
    auto is_word_char = [&valid_chars_set](const char &c) {
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

}  // namespace utility
}  // namespace open3d
