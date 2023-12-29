# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2023 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

"""This script inspects the open3d_tf_ops library and generates function wrappers"""
import os
import sys
import inspect
import argparse
import textwrap
import tensorflow as tf
from yapf.yapflib.yapf_api import FormatFile

INDENT_SPACES = '    '

FN_TEMPLATE_STR = '''
def {fn_name_short}({fn_args}):
{docstring}
    return {fn_name}({args_fwd})

'''


def main():
    parser = argparse.ArgumentParser(description="Creates the ops.py file")
    parser.add_argument("--input",
                        type=str,
                        required=True,
                        help="input file with header")
    parser.add_argument("--output", type=str, required=True, help="output file")
    parser.add_argument("--lib",
                        type=str,
                        required=True,
                        help="path to open3d_tf_ops.so")

    args = parser.parse_args()
    print(args)

    oplib = tf.load_op_library(args.lib)

    generated_function_strs = ''
    for fn_name, value in inspect.getmembers(oplib):
        if not inspect.isfunction(value) or not fn_name.startswith(
                'open3d_') or fn_name.endswith('_eager_fallback'):
            continue

        docstring = getattr(oplib, fn_name).__doc__
        docstring = '"""' + docstring + '\n"""'
        docstring = textwrap.indent(docstring, INDENT_SPACES)

        signature = inspect.signature(value)

        fn_args = []
        args_fwd = []
        for _, param in signature.parameters.items():
            tmp = param.name
            if param.default != inspect.Parameter.empty:
                if isinstance(param.default, str):
                    tmp += '="{}"'.format(str(param.default))
                elif isinstance(param.default, type(tf.float32)):
                    tmp += '=_tf.{}'.format(param.default.name)
                else:
                    tmp += '={}'.format(str(param.default))

            fn_args.append(tmp)
            args_fwd.append('{arg}={arg}'.format(arg=param.name))
        fn_args = ', '.join(fn_args)
        args_fwd = ', '.join(args_fwd)
        generated_function_strs += FN_TEMPLATE_STR.format(
            fn_name_short=fn_name[7:],
            fn_name='_lib.' + fn_name,
            fn_args=fn_args,
            docstring=docstring,
            args_fwd=args_fwd)

    with open(args.input, 'r') as f:
        input_header = f.read()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(input_header + generated_function_strs)
    FormatFile(args.output, in_place=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
