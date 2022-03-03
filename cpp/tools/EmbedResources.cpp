#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "open3d/Open3D.h"

#define LINE_NUMBER_START 17

namespace fs = std::experimental::filesystem;

int main(int argc, char* argv[]) {
    std::ifstream src_in;
    std::cout << argc << std::endl;
    src_in.open(std::string(argv[1]));
    // std::vector<std::pair<std::string, std::streampos>> cpp_line_positions;
    std::string src_data;
    // std::string line;
    // int line_number = 0;
    // while (std::getline(src_in, line)) {
    //     line_number++;
    //     if (line_number == LINE_NUMBER_START - 1) break;
    // }
    // line.clear();
    // while (src_in >> line) {
    //     char brace;
    //     src_in >> brace;
    //     if (line.find("Resource(") == std::string::npos) {
    //         break;
    //     }
    //     std::cout << line.substr(10, line.length() - 12) << std::endl;
    //     const std::streampos position = src_in.tellg();
    //     cpp_line_positions.push_back(
    //             {line.substr(10, line.length() - 12), position});
    //     src_in >> line;
    //     line.clear();
    // }
    // src_in.close();

    if(src_in) {
        std::ostringstream ss;
        ss << src_in.rdbuf();
        src_data = ss.str();
    }
    std::ofstream cpp_out;
    cpp_out.open(std::string(argv[1]), std::ios::trunc);

    // std::cout << std::setw(2);
    // for (auto name_pos =  cpp_line_positions.rbegin(); name_pos != cpp_line_positions.rend(); ++name_pos) {
    //     fs::path resources_dir_path = std::string(argv[2]);
    //     fs::path resource_path = resources_dir_path / name_pos->first;
    //     std::cout << resource_path << std::endl;
    //     std::vector<char> resource_data;
    //     std::string error_str;
    //     if (open3d::utility::filesystem::FReadToBuffer(resource_path, resource_data, &error_str)) {
    //         cpp_out.seekp(name_pos->second);
    //         for (auto& byte : resource_data) {
    //             cpp_out << "0x" << std::hex << (int)(unsigned char) byte << ", ";
    //         }
    //         cpp_out << "})," << std::endl;
    //     }
    // }
    size_t found = src_data.find("if (resource_name == \"");
    fs::path resources_dir_path = std::string(argv[2]);
    while (found != std::string::npos) {
        std::stringstream name_stream;
        name_stream << src_data.substr(found+22);
        std::string resource_name;
        name_stream >> resource_name;
        resource_name = resource_name.substr(0, resource_name.length() - 2);
        std::cout << resource_name << std::endl;
        fs::path resource_path = resources_dir_path / resource_name;
        std::vector<char> resource_data;
        std::string error_str;
        std::vector<char> test_chars; 
        if (open3d::utility::filesystem::FReadToBuffer(resource_path, resource_data, &error_str)) {
            std::stringstream hex_data;
            hex_data << "\n        return {";
            for (auto byte : resource_data) {
                hex_data << (int)byte << ", ";
                test_chars.push_back((int)byte);
            }
            hex_data << "};";
            std::cout << (test_chars == resource_data) << std::endl;
            src_data.insert(found+26+resource_name.length(), hex_data.str());
        }
        found = src_data.find("if (resource_name == \"", found+1);
    }

    cpp_out << src_data;
    cpp_out.close();
    return 0;
}
