#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>

#include "open3d/Open3D.h"

namespace fs = std::experimental::filesystem;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <output_directory>";
        return 0;
    }
    
    fs::path output_file_path = fs::path(argv[1]) / "brightday_ibl.ktx";

    std::ofstream file_out;
    file_out.open(output_file_path.string(), std::ios::trunc);
    for (auto byte : brightday_ibl_ktx()) {
        file_out << byte;
    }
    file_out.close();

    return 0;
}
