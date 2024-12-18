import argparse
import textwrap
import sys
import os
from yapf.yapflib.yapf_api import FormatFile


from paddle.utils.cpp_extension.extension_utils import (
    load_op_meta_info_and_register_op,
    _get_api_inputs_str,
    _gen_output_content
)


def remove_op_name_prefix(op_name):
    PADDLE_OPS_PREFIX = "open3d_"

    assert op_name.startswith(PADDLE_OPS_PREFIX), "Paddle operators should be start with `open3d_`."
    func_name = op_name[len(PADDLE_OPS_PREFIX):]

    return func_name


def custom_api_header():
    HEADER = textwrap.dedent(
        """
        # ----------------------------------------------------------------------------
        # -                        Open3D: www.open3d.org                            -
        # ----------------------------------------------------------------------------
        # The MIT License (MIT)
        #
        # Copyright (c) 2018-2024 www.open3d.org
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

        # This file is machine generated. Do not modify.
        from paddle import _C_ops
        from paddle.framework import in_dynamic_or_pir_mode
        from paddle.base.layer_helper import LayerHelper
        from . import return_types
        """
    ).lstrip()

    return HEADER


def custom_api_footer(custom_ops):
    FOOTER = textwrap.dedent(
        """
        __all__ = [
            {export_func_name_strs}
        ]
        """
    ).lstrip()

    export_func_name_strs = ""
    for op_name in custom_ops:
        export_func_name_strs += f"'{remove_op_name_prefix(op_name)}', \n"

    return FOOTER.format(
        export_func_name_strs = export_func_name_strs
    )


def custom_api_content(op_name):
    (
        params_list,
        ins_map,
        attrs_map,
        outs_list,
        in_names,
        _,
        out_names,
        inplace_reverse_idx,
    ) = _get_api_inputs_str(op_name)
    dynamic_content, static_content = _gen_output_content(
        op_name,
        in_names,
        out_names,
        ins_map,
        attrs_map,
        outs_list,
        inplace_reverse_idx,
    )
    API_TEMPLATE = textwrap.dedent(
        """
        def {func_name}({params_list}):
            # The output variable's dtype use default value 'float32',
            # and the actual dtype of output variable will be inferred in runtime.
            if in_dynamic_or_pir_mode():
                outs = _C_ops._run_custom_op("{op_name}", {params_list})
                {dynamic_content}
            else:
                {static_content}
        """
    ).lstrip()

    # NOTE: Hack return express to wrapper multi return value by return_types
    if len(out_names) > 1:
        RETURN_NAMEDTUPLE_TEMPLATE = textwrap.dedent("""return return_types.{op_name}(*res)""").lstrip()
        REPLACED_RETURN_TEMPLATE = textwrap.dedent("""return res[0] if len(res)==1 else res""").lstrip()
        dynamic_content = dynamic_content.replace(REPLACED_RETURN_TEMPLATE, RETURN_NAMEDTUPLE_TEMPLATE.format(op_name=op_name))
        static_content = static_content.replace(REPLACED_RETURN_TEMPLATE, RETURN_NAMEDTUPLE_TEMPLATE.format(op_name=op_name))

    func_name = remove_op_name_prefix(op_name)

    # generate python api file
    api_content = API_TEMPLATE.format(
        func_name=func_name,
        op_name=op_name,
        params_list=params_list,
        dynamic_content=dynamic_content,
        static_content=static_content,
    )

    NAMEDTUPLE_TEMPLATE= textwrap.dedent("""{op_name} = _namedtuple('{op_name}', '{out_names}')""").lstrip()
    out_names = ' '.join([out_name for out_name in out_names])
    api_namedtuple = NAMEDTUPLE_TEMPLATE.format(
                op_name=op_name, out_names=out_names)


    return api_content, api_namedtuple


def main():
    parser = argparse.ArgumentParser(
        description="Creates the ops.py and return_types.py files")
    parser.add_argument("--input_return_types_py_in",
                        type=str,
                        required=True,
                        help="input file with header")
    parser.add_argument("--lib",
                        type=str,
                        required=True,
                        help="path to open3d_paddle_ops.so")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="output directory")
    args = parser.parse_args()

    generated_fuction_strs = ""
    generated_namedtuple_strs = ""
    custom_ops = load_op_meta_info_and_register_op(args.lib)
    for _custom_op in custom_ops:
        generated_fuction_str, generated_namedtuple_str = custom_api_content(_custom_op)
        generated_fuction_strs += generated_fuction_str + "\n"
        generated_namedtuple_strs += generated_namedtuple_str + "\n"

    CUSTOM_API_TEMPLATE = textwrap.dedent("""
        {custom_api_header}
                                          
        {custom_api_content}
                                          
        {custom_api_footer}
    """).lstrip()
    generated_ops_strs = CUSTOM_API_TEMPLATE.format(
        custom_api_header = custom_api_header(),
        custom_api_content = generated_fuction_strs,
        custom_api_footer = custom_api_footer(custom_ops)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_ops_py_path = os.path.join(args.output_dir, 'ops.py')
    with open(output_ops_py_path,'w') as f:
        f.write(generated_ops_strs)
    FormatFile(output_ops_py_path, in_place=True)

    output_return_types_py_path = os.path.join(args.output_dir,
                                               'return_types.py') 
    with open(args.input_return_types_py_in, 'r') as f:
        input_header = f.read()
    with open(output_return_types_py_path, 'w') as f:
        f.write(input_header + generated_namedtuple_strs)

    return 0


if __name__ == '__main__':
    sys.exit(main())