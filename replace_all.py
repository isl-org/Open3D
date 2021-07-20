from pathlib import Path
import os
from pprint import pprint

pwd = Path(os.path.dirname(os.path.realpath(__file__)))


def get_source_files(root_dir):
    extensions = [".cpp", ".cc", ".h", ".cu", ".cuh"]
    source_files = []
    for extension in extensions:
        source_files.extend(Path(root_dir).glob("**/*" + extension))

    source_files = [str(source_file) for source_file in source_files]
    source_files = [
        source_file for source_file in source_files
        if "Dtype.h" not in source_file
    ]
    source_files = [
        source_file for source_file in source_files
        if "Dtype.cpp" not in source_file
    ]

    return source_files


def replace_string_in_file(file_path, src, dst):
    with open(file_path) as f:
        lines = f.readlines()
        lines = [line.replace(src, dst) for line in lines]

    with open(file_path, "w") as f:
        f.writelines(lines)


if __name__ == '__main__':
    root_dir = pwd / "cpp"
    file_paths = get_source_files(root_dir)
    file_paths = sorted(file_paths)

    # srcs_dsts = [
    #     ("core::Dtype::Undefined", "core::kUndefined"),
    #     ("core::Dtype::Float32", "core::kFloat32"),
    #     ("core::Dtype::Float64", "core::kFloat64"),
    #     ("core::Dtype::Int8", "core::kInt8"),
    #     ("core::Dtype::Int16", "core::kInt16"),
    #     ("core::Dtype::Int32", "core::kInt32"),
    #     ("core::Dtype::Int64", "core::kInt64"),
    #     ("core::Dtype::UInt8", "core::kUInt8"),
    #     ("core::Dtype::UInt16", "core::kUInt16"),
    #     ("core::Dtype::UInt32", "core::kUInt32"),
    #     ("core::Dtype::UInt64", "core::kUInt64"),
    #     ("core::Dtype::Bool", "core::kBool"),
    #     ("Dtype::Undefined", "core::kUndefined"),
    #     ("Dtype::Float32", "core::kFloat32"),
    #     ("Dtype::Float64", "core::kFloat64"),
    #     ("Dtype::Int8", "core::kInt8"),
    #     ("Dtype::Int16", "core::kInt16"),
    #     ("Dtype::Int32", "core::kInt32"),
    #     ("Dtype::Int64", "core::kInt64"),
    #     ("Dtype::UInt8", "core::kUInt8"),
    #     ("Dtype::UInt16", "core::kUInt16"),
    #     ("Dtype::UInt32", "core::kUInt32"),
    #     ("Dtype::UInt64", "core::kUInt64"),
    #     ("Dtype::Bool", "core::kBool"),
    # ]

    srcs_dsts = [
        ("core::kUndefined", "core::Undefined"),
        ("core::kFloat32", "core::Float32"),
        ("core::kFloat64", "core::Float64"),
        ("core::kInt8", "core::Int8"),
        ("core::kInt16", "core::Int16"),
        ("core::kInt32", "core::Int32"),
        ("core::kInt64", "core::Int64"),
        ("core::kUInt8", "core::UInt8"),
        ("core::kUInt16", "core::UInt16"),
        ("core::kUInt32", "core::UInt32"),
        ("core::kUInt64", "core::UInt64"),
        ("core::kBool", "core::Bool"),
    ]

    for src, dst in srcs_dsts:
        for file_path in file_paths:
            replace_string_in_file(file_path, src, dst)
