# stomatasplitter.py

from PIL import Image
import numpy as np
import pandas as pd
import os
import glob
import subprocess

grab_list_file = "grabfile.csv"

grab_df = pd.read_csv(grab_list_file)

print(grab_df.head())

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
    # continue
    # print(row)
    vals = row[1]
    base   = Image.open(vals["base"])
    annot  = Image.open(vals["annot"])
    clumps = pd.read_csv(vals["clumps"])
    
    s = 4
    c = 4

    # print(os.path.splitext(os.path.basename(vals['base']))[0])
    # print(os.path.splitext(os.path.basename(vals['annot']))[0])
    # print(os.path.splitext(os.path.basename(vals['clumps']))[0])

    for clm in clumps.iterrows():
        # print(clm)

        clmvals = clm[1]
        y0, x0, y1, x1 = clmvals["bbox-0"], clmvals["bbox-1"], clmvals["bbox-2"], clmvals["bbox-3"]
        xc, yc = (x0+x1)//2, (y0+y1)//2
        xl, xr, yu, yd = xc-36, xc+36, yc-36, yc+36
        
        savedict = {"Hit":"single", "Cl. Hit":"cluster"}
        numdict = {"Hit":0, "Cl. Hit":1}

        # print(f"Original BBox: x0={clmvals['bbox-0']}, y0={clmvals['bbox-1']}, x1={clmvals['bbox-2']}, y1={clmvals['bbox-3']}")
        # print(f"Parsed BBox: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
        # print(f"Center: xc={xc}, yc={yc}")
        # print(f"Crop Coords: xl={xl}, yu={yu}, xr={xr}, yd={yd}\n-------------------\n")
        # print(clmvals["Notes"])
        # print(vals["clumps"])
        # print(clmvals)
        savetag = savedict[clmvals["Notes"]]
        saveclass = lambda v: "train" if v % 5 else "val"

        clump = base.crop((xl, yu, xr, yd))
        # clump.show()
        clump.save(f"training_folder_clumpcatcher/{saveclass(c if numdict[clmvals['Notes']] else s)}/{savetag}/{os.path.splitext(os.path.basename(vals['base']))[0]}_{clm[0]}_{xc:04d}x_{yc:04d}y.png")

        s +=   numdict[clmvals["Notes"]]
        c += 1-numdict[clmvals["Notes"]]

# for ov in glob.glob("additional_clump_images/clustermarks/*.png"):
#     fn = os.path.splitext(os.path.basename(ov))[0]
#     subprocess.run(f'python clumps_table_SUF.py --input_path={ov} --output_folder=additional_clump_images/clumps/', shell=True)


for cl in glob.glob("additional_clump_images/clumps/*.csv"):
    clfl = pd.read_csv(cl)
    q=4
    # print(cl)
    img = Image.open(f"additional_clump_images/base/{os.path.splitext(os.path.basename(cl))[0]}.tif")
    for p in clfl.iterrows():
        v = p[1]
        y0, x0, y1, x1 = v["bbox-0"], v["bbox-1"], v["bbox-2"], v["bbox-3"]
        # print([v["bbox-0"], v["bbox-1"], v["bbox-2"], v["bbox-3"]])
        xc, yc = (x0+x1)//2, (y0+y1)//2
        xc = int(xc)
        yc = int(yc)
        xl, xr, yu, yd = xc-36, xc+36, yc-36, yc+36
        glump = img.crop((xl, yu, xr, yd))
        # glump.show()
        glump.save(f"training_folder_clumpcatcher/{('train' if q%5 else 'val')}/cluster/{os.path.splitext(os.path.basename(cl))[0]}_{cl[0]}_{xc:04d}x_{yc:04d}y.png")
        q += 1

    
# python training_mbn_is.py --trainingfolder="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/training_folder_clumpcatcher/train" --validationfolder="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/training_folder_clumpcatcher/val"  --outputcsv=testtable_retest.csv  