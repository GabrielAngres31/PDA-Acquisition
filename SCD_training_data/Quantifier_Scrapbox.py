import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob


IMAGE_DIR = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\BASE"
TILE_OUTS = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\tiles"
TILE_MASK_OUTS = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\tile_masks"

IN_FILE = "cot_A.tif"

IMAGE = cv2.imread(IMAGE_DIR + "\\" + IN_FILE)

TILESIZE = 64

CUTOFF_THRESHOLD = 64*64*2

def grab_clip(image, tile_size, coordinates):
    x_coord, y_coord = coordinates
    return image[x_coord:x_coord+tile_size, y_coord:y_coord+tile_size, :]

width, height = IMAGE.shape[0:2]

# for x in tqdm.tqdm(range(width)):
#     for y in tqdm.tqdm(range(height)):


# listogram = [np.sum(grab_clip(IMAGE, TILESIZE, (x, y))) for x in range(width) for y in range(height) if np.sum(grab_clip(IMAGE, TILESIZE, (x, y))) > CUTOFF_THRESHOLD]
# print(f"{len(listogram)} valid tiles out of {(width-TILESIZE)*(height-TILESIZE)} possible tiles || [ {(1 - len(listogram)/((width-TILESIZE)*(height-TILESIZE)))}% ] Reduction")
# print(IMAGE.size)

# histogram = np.histogram(listogram)

# print(histogram)

# plt.hist(listogram, bins = 256) # "768"
# plt.show()


# TODO: FIGURE OUT THE PROPORTION OF MASK IMAGES CONTENTS
mask_file_num = len(glob.glob(TILE_MASK_OUTS+'\\*.tif'))

maskogram = [np.sum(cv2.imread(f)) for f in glob.glob(TILE_MASK_OUTS+"\\*.tif") if np.sum(cv2.imread(f)) > 16*256]

print(f"{len(maskogram)} valid tiles out of {mask_file_num} possible tiles || [ {(1 - len(maskogram)/(mask_file_num))}% ] Reduction")
#print(IMAGE.size)
print(maskogram)
histogram = np.histogram(maskogram)

print(histogram)

plt.hist(maskogram, bins = 4096) # "768"
plt.show()