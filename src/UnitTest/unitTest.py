#!/usr/bin/env python

"""
A simple command line tool for creating test cases from header/source files.
"""

import sys
from pprint import pprint


print sys.argv[1]
name = sys.argv[1].split('.')[0]
# print name

with open(sys.argv[1]) as f:
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

pprint(lines)
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

for line in lines:
    print(template % (name, line)),
print
