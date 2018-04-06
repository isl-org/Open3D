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
    upper = list()
    for c in list(fileName):
        if c.isupper():
            upper.append(c)
            continue
        else:
            if 1 == len(upper):
                label.append('_')
                label.extend(upper)
                upper = list()
                label.append(c.upper())
            elif 1 < len(upper):
                label.append('_')
                label.extend(upper[:-1])
                label.append('_')
                label.append(upper[-1])
                upper = list()
                label.append(c.upper())
            else:
                label.append(c.upper())

    label.append('_H')

    output = "".join(label)

    return output

for header in headers:
    basename = os.path.basename(header)
    guard = label(basename)
    print("%-50s%s" % (basename, guard))
