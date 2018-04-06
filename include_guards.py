#!/usr/bin/env python3

import os
import glob

def headers(path):
    output = list()
    files = glob.glob(path + '/**/*.h', recursive=True)
    for file in files:
        output.append(file)
    return output

path = "/home/dpetre/Open3D/issue_278/src/Core"

files = headers(path)

for file in files:
    basename = os.path.basename(file)
    fileName = os.path.splitext(basename)[0]

    label = list()
    label.extend(list("OPEN3D"))

    prvUpper = False
    for c in list(fileName):
        if c.isupper():
            if not prvUpper and 0 < len(label):
                label.append('_')
            prvUpper = True
        else:
            prvUpper = False
        label.append(c.upper())
    label.append('_H')

    print("%-50s%s" % (fileName, "".join(label)))
