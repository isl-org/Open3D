"""This script inspects the open3d_torch_ops library and generates function wrappers"""
import os
import sys
import inspect
import argparse
import textwrap
import torch
import re
from glob import glob
from collections import namedtuple
from yapf.yapflib.yapf_api import FormatFile

INDENT_SPACES = '    '

FN_TEMPLATE_STR = '''
def {fn_name}({fn_args}):
{docstring}
    return _torch.ops.open3d.{fn_name}({args_fwd})

'''

FN_RETURN_NAMEDTUPLE_TEMPLATE_STR = '''
def {fn_name}({fn_args}):
{docstring}
    return return_types.{fn_name}(*_torch.ops.open3d.{fn_name}({args_fwd}))

'''

NAMEDTUPLE_TEMPLATE_STR = "{name} = _namedtuple( '{name}', '{fields}')\n"


class Argument:
    __slots__ = ['type', 'name', 'default_value']

    def __init__(self, arg_type, name, default_value=None):
        self.type = arg_type
        self.name = name
        self.default_value = default_value


Schema = namedtuple('Schema', ['name', 'arguments', 'returns'])


# not used at the moment because we can use the parser from pytorch 1.4.0 for now.
# just in case keep this for the initial commit
def parse_schema_from_docstring(docstring):
    """Parses the schema from the definition in the docstring of the function.

    At the moment we only allow tuples and a single Tensor as return value.
    All input arguments must have a name.
    E.g. the following are schemas for which we can generate wrappers

    open3d::my_function(int a, Tensor b, Tensor c) -> (Tensor d, Tensor e)
    open3d::my_function(int a, Tensor b, Tensor c) -> Tensor d
    open3d::my_function(int a, Tensor b, str c='bla') -> Tensor d
    """
    m = re.search('with schema: open3d::(.*)$', docstr)
    fn_signature = m.group(1)
    m = re.match('^(.*)\((.*)\) -> (.*)', fn_signature)
    fn_name, arguments, returns = m.group(1), m.group(2), m.group(3)
    arguments = [tuple(x.strip().split(' ')) for x in arguments.split(',')]
    arguments = [Argument(x[0], *x[1].split('=')) for x in arguments]
    # torch encodes str default values as octals
    # -> convert a string that contains octals to a proper python str
    for a in arguments:
        if not a.default_value is None and a.typename == 'str':
            a.default_value = bytes([
                int(x, 8) for x in a.default_value[1:-1].split('\\')[1:]
            ]).decode('utf-8')

    if returns.strip().startswith('('):
        # remove tuple parenthesis
        returns = returns.strip()[1:-1]

    returns = [tuple(x.strip().split(' ')) for x in returns.split(',')]
    return Schema(fn_name, arguments, returns)


def get_tensorflow_docstring_from_file(path):
    """Extracts the docstring from a tensorflow register op file"""
    if path is None:
        return ""
    with open(path, 'r') as f:
        tf_reg_op_file = f.read()
    # docstring must use raw string with R"doc( ... )doc"
    m = re.search('R"doc\((.*?)\)doc"',
                  tf_reg_op_file,
                  flags=re.MULTILINE | re.DOTALL)
    return m.group(1).strip()


def find_op_reg_file(ops_dir, op_name):
    """Tries to find the corresponding tensorflow file for the op_name"""
    lowercase_filename = op_name.replace('_', '') + 'ops.cpp'
    print(lowercase_filename)
    all_op_files = glob(os.path.join(ops_dir, '**', '*Ops.cpp'), recursive=True)
    op_file_dict = {os.path.basename(x).lower(): x for x in all_op_files}
    if lowercase_filename in op_file_dict:
        return op_file_dict[lowercase_filename]
    else:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Creates the ops.py and return_types.py files")
    parser.add_argument("--input_ops_py_in",
                        type=str,
                        required=True,
                        help="input file with header")
    parser.add_argument("--input_return_types_py_in",
                        type=str,
                        required=True,
                        help="input file with header")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="output directory")
    parser.add_argument("--lib",
                        type=str,
                        required=True,
                        help="path to open3d_torch_ops.so")
    parser.add_argument("--tensorflow_ops_dir",
                        type=str,
                        required=True,
                        help="This is cpp/open3d/ml/tensorflow")

    args = parser.parse_args()
    print(args)

    torch.ops.load_library(args.lib)

    generated_function_strs = ''
    generated_namedtuple_strs = ''
    for schema in torch._C._jit_get_all_schemas():
        if not schema.name.startswith('open3d::'):
            continue

        docstring = get_tensorflow_docstring_from_file(
            find_op_reg_file(args.tensorflow_ops_dir, schema.name[8:]))
        if docstring:
            docstring = '"""' + docstring + '\n"""'
            docstring = textwrap.indent(docstring, INDENT_SPACES)

        fn_args = []
        args_fwd = []
        for arg in schema.arguments:
            tmp = arg.name
            if not arg.default_value is None:
                if isinstance(arg.default_value, str):
                    tmp += '="{}"'.format(str(arg.default_value))
                else:
                    tmp += '={}'.format(str(arg.default_value))

            fn_args.append(tmp)
            args_fwd.append('{arg}={arg}'.format(arg=arg.name))
        fn_args = ', '.join(fn_args)
        args_fwd = ', '.join(args_fwd)

        if len(schema.returns) > 1:
            template_str = FN_RETURN_NAMEDTUPLE_TEMPLATE_STR
            fields = ' '.join([x.name for x in schema.returns])
            generated_namedtuple_strs += NAMEDTUPLE_TEMPLATE_STR.format(
                name=schema.name[8:], fields=fields)
        else:
            template_str = FN_TEMPLATE_STR

        generated_function_strs += template_str.format(
            fn_name=schema.name[8:],  # remove the 'open3d::'
            fn_args=fn_args,
            docstring=docstring,
            args_fwd=args_fwd)

    with open(args.input_ops_py_in, 'r') as f:
        input_header = f.read()

    os.makedirs(args.output_dir, exist_ok=True)
    output_ops_py_path = os.path.join(args.output_dir, 'ops.py')
    with open(output_ops_py_path, 'w') as f:
        f.write(input_header + generated_function_strs)
    FormatFile(output_ops_py_path, in_place=True)

    output_return_types_py_path = os.path.join(args.output_dir,
                                               'return_types.py')
    with open(args.input_return_types_py_in, 'r') as f:
        input_header = f.read()
    with open(output_return_types_py_path, 'w') as f:
        f.write(input_header + generated_namedtuple_strs)
    FormatFile(output_return_types_py_path, in_place=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
