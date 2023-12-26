// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/visualization/gui/Util.h"

#include <cmath>

#include "open3d/utility/FileSystem.h"
#include "open3d/visualization/gui/Color.h"

namespace open3d {
namespace visualization {
namespace gui {

ImVec4 colorToImgui(const Color &color) {
    return ImVec4(color.GetRed(), color.GetGreen(), color.GetBlue(),
                  color.GetAlpha());
}

uint32_t colorToImguiRGBA(const Color &color) {
    return IM_COL32(int(std::round(255.0f * color.GetRed())),
                    int(std::round(255.0f * color.GetGreen())),
                    int(std::round(255.0f * color.GetBlue())),
                    int(std::round(255.0f * color.GetAlpha())));
}

std::string FindFontPath(std::string font, FontStyle style) {
    using namespace open3d::utility::filesystem;

    std::vector<std::string> kNormalSuffixes = {
            " Regular.ttf", " Regular.ttc", " Regular.otf", " Normal.ttf",
            " Normal.ttc",  " Normal.otf",  " Medium.ttf",  " Medium.ttc",
            " Medium.otf",  " Narrow.ttf",  " Narrow.ttc",  " Narrow.otf",
            "-Regular.ttf", "-Regular.ttc", "-Regular.otf", "-Normal.ttf",
            "-Normal.ttc",  "-Normal.otf",  "-Medium.ttf",  "-Medium.ttc",
            "-Medium.otf",  "-Narrow.ttf",  "-Narrow.ttc",  "-Narrow.otf",
            "Regular.ttf",  "-Regular.ttc", "-Regular.otf", "Normal.ttf",
            "Normal.ttc",   "Normal.otf",   "Medium.ttf",   "Medium.ttc",
            "Medium.otf",   "Narrow.ttf",   "Narrow.ttc",   "Narrow.otf"};

    std::vector<std::string> suffixes;
    switch (style) {
        case FontStyle::NORMAL:
            suffixes = kNormalSuffixes;
            break;
        case FontStyle::BOLD:
            suffixes = {
                " Bold.ttf",
                " Bold.ttc",
                " Bold.otf",
                "-Bold.ttf",
                "-Bold.ttc",
                "-Bold.otf",
                "Bold.ttf",
                "Bold.ttc",
                "Bold.oft"
#if _WIN32
                ,
                "b.ttf",
                "b.ttc",
                "b.otf"
#endif  // _WIN32
            };
            break;
        case FontStyle::ITALIC:
            suffixes = {
                " Italic.ttf",
                " Italic.ttc",
                " Italic.otf",
                "-Italic.ttf",
                "-Italic.ttc",
                "-Italic.otf",
                "Italic.ttf",
                "Italic.ttc",
                "Italic.otf",
                "-MediumItalic.ttf",
                "-MediumItalic.ttc",
                "-MediumItalic.otf",
                "MediumItalic.ttf",
                "MediumItalic.ttc",
                "MediumItalic.otf",
                "-Oblique.ttf",
                "-Oblique.ttc",
                "-Oblique.otf",
                "Oblique.ttf",
                "Oblique.ttc",
                "Oblique.otf",
                "-MediumOblique.ttf",
                "-MediumOblique.ttc",
                "-MediumOblique.otf",
                "MediumOblique.ttf",
                "MediumOblique.ttc",
                "MediumOblique.otf"
#if _WIN32
                ,
                "i.ttf",
                "i.ttc",
                "i.otf"
#endif  // _WIN32
            };
            break;
        case FontStyle::BOLD_ITALIC:
            suffixes = {
                " Bold Italic.ttf",
                " Bold Italic.ttc",
                " Bold Italic.otf",
                "-BoldItalic.ttf",
                "-BoldItalic.ttc",
                "-BoldItalic.otf",
                "BoldItalic.ttf",
                "BoldItalic.ttc",
                "BoldItalic.oft"
#if _WIN32
                ,
                "bi.ttf",
                "bi.ttc",
                "bi.otf"
#endif  // _WIN32
            };
            break;
    }

    if (FileExists(font)) {
        if (style == FontStyle::NORMAL) {
            return font;
        } else {
            // The user provided an actual font file, not just a
            // font name. Since we are looking for bold and/or italic,
            // we need to "stem" the font file and attempt to look for
            // the bold and/or italicized versions.
            for (auto &suf : kNormalSuffixes) {
                if (font.rfind(suf) != std::string::npos) {
                    font = font.substr(0, font.size() - suf.size());
                    break;
                }
            }
            // The font name doesn't have any of the suffixes,
            // so just remove the extension
            font = font.substr(0, font.size() - 4);

            // Check if any of the stylized suffixes work
            for (auto &suf : suffixes) {
                std::string candidate = font + suf;
                if (FileExists(candidate)) {
                    return candidate;
                }
            }
            // Otherwise fail
            return "";
        }
    }

    std::string home;
    char *raw_home = getenv("HOME");
    if (raw_home) {  // std::string(nullptr) is undefined
        home = raw_home;
    }
    std::vector<std::string> system_font_paths = {
#ifdef __APPLE__
            "/System/Library/Fonts", "/Library/Fonts", home + "/Library/Fonts"
#elif _WIN32
            "c:/Windows/Fonts"
#else
            "/usr/share/fonts",
            home + "/.fonts",
#endif  // __APPLE__
    };

#ifdef __APPLE__
    std::vector<std::string> font_ext = {".ttf", ".ttc", ".otf"};
    for (auto &font_path : system_font_paths) {
        if (style == FontStyle::NORMAL) {
            for (auto &ext : font_ext) {
                std::string candidate = font_path + "/" + font + ext;
                if (FileExists(candidate)) {
                    return candidate;
                }
            }
        }
    }
    for (auto &font_path : system_font_paths) {
        for (auto &suf : suffixes) {
            std::string candidate = font_path + "/" + font + suf;
            if (FileExists(candidate)) {
                return candidate;
            }
        }
    }
    return "";
#else
    std::string font_ttf = font + ".ttf";
    std::string font_ttc = font + ".ttc";
    std::string font_otf = font + ".otf";
    auto is_match = [font, &font_ttf, &font_ttc,
                     &font_otf](const std::string &path) {
        auto filename = GetFileNameWithoutDirectory(path);
        auto ext = GetFileExtensionInLowerCase(filename);
        if (ext != "ttf" && ext != "ttc" && ext != "otf") {
            return false;
        }
        if (filename == font_ttf || filename == font_ttc ||
            filename == font_otf) {
            return true;
        }
        if (filename.find(font) == 0) {
            return true;
        }
        return false;
    };

    for (auto &font_dir : system_font_paths) {
        auto matches = FindFilesRecursively(font_dir, is_match);
        if (style == FontStyle::NORMAL) {
            for (auto &m : matches) {
                if (GetFileNameWithoutExtension(
                            GetFileNameWithoutDirectory(m)) == font) {
                    return m;
                }
            }
        }
        for (auto &m : matches) {
            auto dir = GetFileParentDirectory(m);  // has trailing slash
            for (auto &suf : suffixes) {
                std::string candidate = dir + font + suf;
                if (m == candidate) {
                    return candidate;
                }
            }
        }
    }
    return "";
#endif  // __APPLE__
}

}  // namespace gui
}  // namespace visualization
}  // namespace open3d
