from pathlib import Path
import os
from pprint import pprint

pwd = Path(os.path.dirname(os.path.realpath(__file__)))


def get_source_files(root_dir):
    extensions = [".py"]
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
    root_dir = pwd / "python"
    file_paths = get_source_files(root_dir)
    file_paths = sorted(file_paths)

    srcs_dsts = [
        ("o3d.core.Dtype.Undefined", "o3d.core.undefined"),
        ("o3d.core.Dtype.Float32", "o3d.core.float32"),
        ("o3d.core.Dtype.Float64", "o3d.core.float64"),
        ("o3d.core.Dtype.Int8", "o3d.core.int8"),
        ("o3d.core.Dtype.Int16", "o3d.core.int16"),
        ("o3d.core.Dtype.Int32", "o3d.core.int32"),
        ("o3d.core.Dtype.Int64", "o3d.core.int64"),
        ("o3d.core.Dtype.UInt8", "o3d.core.uint8"),
        ("o3d.core.Dtype.UInt16", "o3d.core.uint16"),
        ("o3d.core.Dtype.UInt32", "o3d.core.uint32"),
        ("o3d.core.Dtype.UInt64", "o3d.core.uint64"),
        ("o3d.core.Dtype.Bool", "o3d.core.bool"),
        ("o3c.Dtype.Undefined", "o3c.undefined"),
        ("o3c.Dtype.Float32", "o3c.float32"),
        ("o3c.Dtype.Float64", "o3c.float64"),
        ("o3c.Dtype.Int8", "o3c.int8"),
        ("o3c.Dtype.Int16", "o3c.int16"),
        ("o3c.Dtype.Int32", "o3c.int32"),
        ("o3c.Dtype.Int64", "o3c.int64"),
        ("o3c.Dtype.UInt8", "o3c.uint8"),
        ("o3c.Dtype.UInt16", "o3c.uint16"),
        ("o3c.Dtype.UInt32", "o3c.uint32"),
        ("o3c.Dtype.UInt64", "o3c.uint64"),
        ("o3c.Dtype.Bool", "o3c.bool"),
    ]

    for src, dst in srcs_dsts:
        for file_path in file_paths:
            replace_string_in_file(file_path, src, dst)
