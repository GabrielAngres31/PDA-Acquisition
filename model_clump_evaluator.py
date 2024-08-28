# model_clump_evaluator.py

import pandas as pd
import argparse
import src.data
import tqdm
import numpy as np
import PIL
from pathlib import Path
import matplotlib.pyplot as plt

import skimage.measure as skimm
import skimage.morphology as skimorph

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage


files = [f'reference_figures/_test4_{i:03}.png' for i in range(0, 30)]
combine = skimage.io.imread(files[0])

for i in files[1:]:
    image = skimage.io.imread(i).astype(int)
    combine = np.add(combine, image)
plt.imshow(combine) #, cmap=plt.cm.gray)
plt.show()
plt.imsave("reference_figures/_test4_all_outliner.png")

