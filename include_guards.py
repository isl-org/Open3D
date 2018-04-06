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

files1 = list()
for (dirpath, dirnames, filenames) in os.walk(path):
    for file in filenames:
        if file.endswith('.h'):
            # print os.path.join(dirpath, file)
            files1.append(os.path.join(dirpath, file))



files2 = headers(path)

print(len(files1))
print(len(files2))

# compare the two methods
# for i in range(len(files1)):
#     print("%s\t%s" % (files1[i] == files2[i], files2[i]))

for file in files2:
    filename = os.path.basename(file)
    name = os.path.splitext(filename)[0]
    nameList2 = list()
    nameList2.extend(list("OPEN3D"))
    index = 0
    prvUpper = False
    for c in list(name):
        if c.isupper():
            if not prvUpper and 0 < len(nameList2):
                nameList2.append('_')
            prvUpper = True
        else:
            prvUpper = False
        nameList2.append(c.upper())
    nameList2.append('_H')
    # print(nameList2)
    print("%-50s%s" % (name, "".join(nameList2)))

    # for item in nameList2:
    #     newName = ''.join(item)
    #     print(newName)
