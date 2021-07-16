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

#pragma once

#include <string>
#include <vector>

#include "open3d/visualization/gui/Gui.h"

namespace open3d {
namespace visualization {
namespace gui {

class FontDescription {
public:
    constexpr static const char *SANS_SERIF = "sans-serif";
    constexpr static const char *MONOSPACE = "monospace";

    /// Creates a font description. This must be passed to
    /// Application::AddFont() before the window is created.
    /// \param typeface A path to a TrueType (.ttf), TrueType Collection (.ttc),
    ///                 or OpenType (.otf) file, or it is the name of the font,
    ///                 in which case the system font paths will be searched
    ///                 to find the font file. This typeface will be used for
    ///                 roman characters (Extended Latin, that is, European
    ///                 languages).
    /// \param style Ignored if typeface is a file, but will be used to qualify
    ///              the font if a system font name is used. Will be applied to
    ///              any additional language or code point typefaces.
    /// \param point_size The point size (NOT pixel size) of the font. Will be
    ///                   applied to any additional language or code point
    ///                   typefaces. A size of 0 indicates the default size.
    explicit FontDescription(const char *typeface,
                             FontStyle style = FontStyle::NORMAL,
                             int point_size = 0);

    /// Adds code points outside Extended Latin from the specified typeface.
    /// Supported languages are:
    ///   "en" (English)  [this was already added in the constructor]
    ///   "ja" (Japanese)
    ///   "ko" (Korean)
    ///   "th" (Thai)
    ///   "vi" (Vietnamese)
    ///   "zh" (Chinese, 2500 most common characters, 50 MB per window)
    ///   "zh_all" (Chinese, all characters, ~200 MB per window)
    /// All other languages will be assumed to be Cyrillic.
    /// Note that generally fonts do not have CJK glyphs unless they are
    /// specifically a CJK font, although operating systems generally use a
    /// CJK font for you. We do not have the information necessary to do this,
    /// so you will need to provide a font that has the glyphs you need.
    /// In particular, common fonts like "Arial", "Helvetica", and SANS_SERIF
    /// do not contain CJK glyphs.
    void AddTypefaceForLanguage(const char *typeface, const char *lang);

    /// Adds specific code points from the typeface. This is useful for
    /// selectively adding glyphs, for example, from an icon font.
    void AddTypefaceForCodePoints(const char *typeface,
                                  const std::vector<uint32_t> &code_points);

public:  // for internal use, use functions above to set
    struct CPRange {
        std::string path;
        std::string lang;
        std::vector<uint32_t> code_points;  // empty if lang is not ""
    };
    std::vector<CPRange> ranges_;
    FontStyle style_;
    int point_size_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
