# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d

if __name__ == "__main__":
    img_data = o3d.data.JuneauImage()
    print(f"Reading image from file: Juneau.jpg stored at {img_data.path}")
    img = o3d.io.read_image(img_data.path)
    print(img)
    print("Saving image to file: copy_of_Juneau.jpg")
    o3d.io.write_image("copy_of_Juneau.jpg", img)
