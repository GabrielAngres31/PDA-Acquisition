# stomatasplitter.py

"""

This script was used to parse stomata-centered sections from whole images in order to train the clump-detecting network.
It automatically sorted the stomata into a 80/20 training/testing split.
Requires annotated images as an input (in grabfile.csv)
"""

import glob
import os

import pandas as pd
from PIL import Image

grab_list_file = "grabfile.csv"

grab_df = pd.read_csv(grab_list_file)

print(grab_df.head())

# Whether or not to clear the entire folder of training/testing images if these images exist.
# Used to ensure that enormous file folders worth of images aren't duplicated unecessarily.
destroy = True

if destroy:
    for dirpath, dirnames, filenames in os.walk("training_folder_clumpcatcher"):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                os.remove(file_path)
                # print(f"Removed: {file_path}")
            except OSError as e:
                print(f"Error removing {file_path}: {e}")


for row in grab_df.iterrows():
    # Acquire base, annot images and corresponding clumps table
    vals = row[1]
    base = Image.open(vals["base"])
    annot = Image.open(vals["annot"])
    clumps = pd.read_csv(vals["clumps"])

    # These numbers are set to 4, given that the training/testing split is 80/20, so that there is always one training and one testing stomata to start with.
    s = 4
    c = 4

    for clm in clumps.iterrows():
        # Acquire the clump
        clmvals = clm[1]
        y0, x0, y1, x1 = (
            clmvals["bbox-0"],
            clmvals["bbox-1"],
            clmvals["bbox-2"],
            clmvals["bbox-3"],
        )

        # Acquire image bounding box for copying
        xc, yc = (x0 + x1) // 2, (y0 + y1) // 2
        xl, xr, yu, yd = xc - 36, xc + 36, yc - 36, yc + 36

        savedict = {"Hit": "single", "Cl. Hit": "cluster"}
        numdict = {"Hit": 0, "Cl. Hit": 1}

        savetag = savedict[clmvals["Notes"]]

        def saveclass(v):
            return "train" if v % 5 else "val"

        clump = base.crop((xl, yu, xr, yd))
        clump.save(
            f"training_folder_clumpcatcher/{saveclass(c if numdict[clmvals['Notes']] else s)}/{savetag}/{os.path.splitext(os.path.basename(vals['base']))[0]}_{clm[0]}_{xc:04d}x_{yc:04d}y.png"
        )

        s += numdict[clmvals["Notes"]]
        c += 1 - numdict[clmvals["Notes"]]


# Additional loop with the same logic as above called on some additional file sets/csvs to augment the training data after the initial training rounds.
for cl in glob.glob("additional_clump_images/clumps/*.csv"):
    clfl = pd.read_csv(cl)
    q = 4
    img = Image.open(
        f"additional_clump_images/base/{os.path.splitext(os.path.basename(cl))[0]}.tif"
    )
    for p in clfl.iterrows():
        v = p[1]
        y0, x0, y1, x1 = v["bbox-0"], v["bbox-1"], v["bbox-2"], v["bbox-3"]
        xc, yc = (x0 + x1) // 2, (y0 + y1) // 2
        xc = int(xc)
        yc = int(yc)
        xl, xr, yu, yd = xc - 36, xc + 36, yc - 36, yc + 36
        glump = img.crop((xl, yu, xr, yd))
        glump.save(
            f"training_folder_clumpcatcher/{('train' if q % 5 else 'val')}/cluster/{os.path.splitext(os.path.basename(cl))[0]}_{cl[0]}_{xc:04d}x_{yc:04d}y.png"
        )
        q += 1
