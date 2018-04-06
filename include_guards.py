#!/usr/bin/env python3

import os
import glob

# recursively list all the files with a specific extension
def files(path, ext):
    output = glob.glob(path + '/**/*.' + ext, recursive=True)
    return output

path = "/home/dpetre/Open3D/issue_278/src/Core"

headers = files(path, 'h')

def label(header):
    basename = os.path.basename(header)
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

    output = "".join(label)

    return output

for header in headers:
    basename = os.path.basename(header)
    guard = label(basename)
    print("%-50s%s" % (basename, guard))
