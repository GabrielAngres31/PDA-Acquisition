# skimage_outline_scrapbox

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import skimage
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, area_closing, area_opening
from skimage.color import label2rgb
import PIL
import numpy as np

from skimage.filters import try_all_threshold

test_image_path =  "inference/cot6.tif.output.png"
image_in = skimage.io.imread(test_image_path)

otsu_trsh_num = threshold_otsu(image_in)
otsu_fill = area_closing(image_in > otsu_trsh_num, connectivity = square(3), area_threshold = 2500) # Hardcoded dark patch value
otsu_invt = skimage.util.invert(image_in > otsu_trsh_num)

inners = np.logical_and(otsu_fill, otsu_invt)
otsu_clr = area_opening(inners, area_threshold = 200)

skimage.io.imshow(otsu_clr)
plt.show()



# EVAL_FINISH = closing(EVAL_IN, square(3))
# fig, ax = try_all_threshold(image_in, figsize=(10, 8), verbose=False)
# plt.show()



# Remove islands
# Get available circles

