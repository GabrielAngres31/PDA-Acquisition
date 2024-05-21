import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import sys
import tqdm

IMAGE_DIR = ".\\PDA-Acquisition\\SCD_training_data\\source_images\\BASE"
TILE_OUTS = ".\\PDA-Acquisition\\SCD_training_data\\source_images\\tiles"
TILE_MASK_OUTS = ".\\PDA-Acquisition\\SCD_training_data\\source_images\\tile_masks"

IN_FILE = "AI_MASK_V4H4_cot1.tif"
# IN_FILE = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\ANNOTATION\\cot1_STOMATA_MASKS.tiff"

IMAGE = cv2.imread(IMAGE_DIR + "\\" + IN_FILE)
# IMAGE = cv2.imread(IN_FILE)

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

# FILTERED

# TODO: FIGURE OUT THE PROPORTION OF MASK IMAGES CONTENTS
# mask_file_num = len(glob.glob(TILE_MASK_OUTS+'\\*.tif'))

# maskogram = [np.sum(cv2.imread(f)) for f in glob.glob(TILE_MASK_OUTS+"\\*.tif") if np.sum(cv2.imread(f)) > 16*256]

# print(f"{len(maskogram)} valid tiles out of {mask_file_num} possible tiles || [ {(1 - len(maskogram)/(mask_file_num))}% ] Reduction")
# #print(IMAGE.size)
# print(maskogram)
# histogram = np.histogram(maskogram)

# print(histogram)

# plt.hist(maskogram, bins = 4096) # "768"
# plt.show()

# PX by PX

# listogram = [i for i in np.ndarray.flatten(IMAGE[:,:,0]) if i]


# #print(f"{len(listogram)} valid tiles out of {(width-TILESIZE)*(height-TILESIZE)} possible tiles || [ {(1 - len(listogram)/((width-TILESIZE)*(height-TILESIZE)))}% ] Reduction")
# #print(IMAGE.size)

# histogram = np.histogram(listogram)

# print(histogram)

# plt.hist(listogram, bins = 256) # "768"
# plt.show()


# ----

# BLOB DETECTOR

sys.setrecursionlimit(10000)

def is_valid_move(image, visited, row, col, threshold_dimness):
    rows = len(image)
    cols = len(image[0])
    return row >= 0 and row < rows and col >= 0 and col < cols and image[row][col][0] > threshold_dimness and not visited[row][col]

def dfs(image, visited, row, col, blob):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
    visited[row][col] = True
    blob.append((row, col))
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if is_valid_move(image, visited, new_row, new_col, 32):
            dfs(image, visited, new_row, new_col, blob)

def find_blobs(image, threshold_brightness):
    rows = len(image)
    cols = len(image[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    blobs = []
    for i in range(rows):
        for j in range(cols):
            if image[i][j][0] > threshold_brightness and not visited[i][j]:
                blob = []
                dfs(image, visited, i, j, blob)
                blobs.append(blob)
    return blobs

# Example usage:
image = IMAGE

# blobs = find_blobs(image, 220)
# print("Number of blobs:", len(blobs))

bloblist = []
for i in tqdm.tqdm(range(200, 256)):
    bloblist.append(f"{i}: {len(find_blobs(image, i))}")
print(bloblist)

# TODO
# Filter blobs that have greater than certain dimensions on any axis. Go with 64x64.