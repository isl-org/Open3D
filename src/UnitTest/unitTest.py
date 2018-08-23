#!/usr/bin/env python

"""
A simple command line tool for creating test cases from header/source files.
"""

from argparse import ArgumentParser
import os
from pprint import pprint
import sys


def build_argparser():
    parser = ArgumentParser(description = 'A simple command line tool for creating test cases from header/source files.')
    parser.add_argument("-f",  "--file", help = "Path to a .h file.",                   type = str, required = True)
    parser.add_argument("-s",  "--show", help = "Show the unit test templates or not.", type = int, required = False, default = 1)

    return parser

args = build_argparser().parse_args()

# print(dir(args))

filename = os.path.basename(args.file)

name = filename.split('.')[0]
print name
print

with open(args.name) as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]
lines = [line for line in lines if line != '']
lines = [line for line in lines if '//' not in line]
lines = [line for line in lines if '(' in line]

# pprint(lines)
# print

output = []
for line in lines:
    elements = [elem for elem in line.split() if '(' in elem]
    output.append(elements[0])

# pprint(output)
# print

lines = []
[lines.append(line) for line in output if line not in lines]

lines = [line.strip('*') for line in lines]
lines = [line for line in lines if '.' not in line]

output = []
for line in lines:
    elements = [elem for elem in line.split('(')]

    line = elements[0]
    if name == line:
        line = "Constructor"

    if '~' in elements[0]:
        line = "Destructor"

    output.append(line)

lines = output

lines = []
[lines.append(line) for line in output if line not in lines]
lines = [line for line in lines if line != '']
lines = [line for line in lines if line[0].isupper()]
lines = [line for line in lines if ':' not in line]

for line in lines:
    print(line)
print

template = "\n\
// ----------------------------------------------------------------------------\n\
// \n\
// ----------------------------------------------------------------------------\n\
TEST(%s, %s)\n\
{\n\
    NotImplemented();\n\
}\n\
"

if args.show:
    for line in lines:
        print(template % (name, line)),
    print
