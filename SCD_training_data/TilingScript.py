# Imports
import segmentation_models_pytorch as smp
import cv2
import sys
import os
import albumentations as albu
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset as BaseDataset
import glob
import re
import tqdm

IMAGE_DIR = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\BASE"
TILE_OUTS = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\tiles"
TILE_MASK_OUTS = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\tile_masks"

# IN_FILE = "cot_A.tif"
IN_FILE = "cot1.tif"

print("Reading File...")

IMAGE = cv2.imread(IMAGE_DIR + "\\" + IN_FILE)

# GET THE FUNCTION FROM TQDM SCRAPBOX
# [cv2.imwrite(TILE_OUTS + "\\" + IN_FILE[:-4] + f"_{tqdm_comp(i_token)}.tif", grab_clip(CLIP_IMAGE, TILESIZE, tqdm_comp(i_token))) for i_token in tqdm.tqdm([{"counter" : i} for i in tile_corners])]
# [cv2.imwrite(TILE_MASK_OUTS + f"\\{n_token}", classify(test_dataset[tqdm_comp(n_token)])) for n_token in tqdm.tqdm([{"counter" : i} for i in test_dataset.ids])]




TILESIZE = 64

VERT_SPACING = 4
HORZ_SPACING = 4

VERT_OVERLAP = TILESIZE - VERT_SPACING
HORZ_OVERLAP = TILESIZE - HORZ_SPACING



assert HORZ_OVERLAP < TILESIZE and VERT_OVERLAP < TILESIZE, f"Overlap values do not permit image processing! Overlap must be less than current tilesize: {TILESIZE}"

CUTOFF_THRESHOLD = 64*64*2 # Tested

SWITCH_harvestTiles = False
SWITCH_makeMasks = False

# Clip the image down
width, height = IMAGE.shape[0:2]
width_buffer = (width - TILESIZE) % (HORZ_SPACING)
height_buffer = (height - TILESIZE) % (VERT_SPACING)
#clip_width, clip_height = TILESIZE

print("Diagnostic")
print(f"{width} - {TILESIZE} % {HORZ_SPACING} = {width_buffer}")
print(f"{height} - {TILESIZE} % {VERT_SPACING} = {height_buffer}")

print("-----")

print(width_buffer)
print(height_buffer)

CLIP_IMAGE = IMAGE[:width-width_buffer+1, :height-height_buffer+1, :]
#print(CLIP_IMAGE.shape)

clip_width, clip_height = CLIP_IMAGE.shape[0:2]
print(IMAGE.shape)
print(CLIP_IMAGE.shape)

# Generate all tile indices.
print("Generating Tile Indices...")
tile_column_indices = range(0, clip_width, HORZ_SPACING) #TILESIZE-HORZ_OVERLAP
print(tile_column_indices)
tile_row_indices = range(0, clip_height, VERT_SPACING) #TILESIZE-VERT_OVERLAP
print(tile_row_indices)

tile_corners = [(c, r) for c in tile_column_indices for r in tile_row_indices]

print(tile_corners)
print(len(tile_corners))

# Calculate the number of tiles that will be generated.
#print(len(tile_corners))

# Build Tile Dataset

def grab_clip(image, tile_size, coordinates):
    x_coord, y_coord = coordinates
    return image[x_coord:x_coord+tile_size, y_coord:y_coord+tile_size, :]

if SWITCH_harvestTiles:
    print("Clearing Existing TILE Files...")
    [os.remove(f) for f in glob.glob(TILE_OUTS + "\\*.tif")]
    print(f"Generating {len(tile_corners)} Tiles with {HORZ_OVERLAP} <> and {VERT_OVERLAP} /|/...")
    [cv2.imwrite(TILE_OUTS + "\\" + IN_FILE[:-4] + f"_{i}.tif", grab_clip(CLIP_IMAGE, TILESIZE, i)) for i in tile_corners if np.sum(grab_clip(CLIP_IMAGE, TILESIZE, i)) > CUTOFF_THRESHOLD]

# Build Classifier

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

DATA_DIR = TILE_OUTS

#scan_dir = os.path.join(DATA_DIR, 'train')

scan_dir = os.path.join(DATA_DIR)


ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['stomata']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

print("Loading Segmentation Model...")
# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

best_model = torch.load('C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\best_model.pth') #best_model_on_whole.pth')
# create test dataset

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

class Dataset_InUse(BaseDataset):

    CLASSES = ['background', 'stomata']

    def __init__(
            self,
            images_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        #print(self.ids.index(i) if isinstance(i, str) else i)
        image_gry = cv2.imread(self.images_fps[self.ids.index(i) if isinstance(i, str) else i], cv2.IMREAD_GRAYSCALE)

        image_rgb = np.stack((image_gry,)*3, axis = -1)

        if self.augmentation:
            sample = self.augmentation(image=image_rgb)
            image_rgb = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image_rgb)
            image_rgb = sample['image']

        return image_rgb

    def __len__(self):
        return len(self.ids)

def get_preprocessing(preprocessing_fn):

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)



def get_validation_augmentation():
    return albu.Compose([albu.PadIfNeeded(64, 64)])

test_dataset = Dataset_InUse(
    scan_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)    

test_dataset_vis = Dataset_InUse(
    scan_dir, 
    classes=CLASSES,
)

def classify(image_in):
    global best_model

    scan_tensor = torch.from_numpy(image_in).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(scan_tensor)
    pr_mask = np.uint8((pr_mask.squeeze().cpu().numpy().round())*255)
    #print(pr_mask)
    return pr_mask



if SWITCH_makeMasks:
    print("Clearing Existing MASK Files...")
    [os.remove(f) for f in glob.glob(TILE_MASK_OUTS + "\\*.tif")]
    print("Generating Predictions...")
    [cv2.imwrite(TILE_MASK_OUTS + f"\\{n}", classify(test_dataset[n])) for n in test_dataset.ids]
    #[print(classify(test_dataset[n])) for n in test_dataset.ids]

FULL_MASK_CANVAS = np.zeros(IMAGE.shape)

def replaceAtCoordinates(array, chunk, y_coord, x_coord):
    #print(chunk.shape)
    array[int(y_coord):int(y_coord)+chunk.shape[0], int(x_coord):int(x_coord)+chunk.shape[1]] = chunk

def addAtCoordinates(array, chunk, y_coord, x_coord):
    target_chunk = array[int(y_coord):int(y_coord)+chunk.shape[0], int(x_coord):int(x_coord)+chunk.shape[1]]
    # print(chunk.size)
    # print(TILESIZE**2)
    #print(target_chunk.shape)
    if target_chunk.shape != (TILESIZE, TILESIZE, 3):
        #print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEAYEH")
        add_chunk = np.full(target_chunk.shape, 0)
    #target_chunk = array[int(y_coord):int(y_coord)+chunk.shape[0], int(x_coord):int(x_coord)+chunk.shape[1]]
    else:
        denumberer = np.vectorize(lambda n: 1 if n else 0)
        add_chunk = denumberer(chunk)
        #print(max(add_chunk) if max(add_chunk) else "")
        #print(add_chunk.shape)
    array[int(y_coord):int(y_coord)+chunk.shape[0], int(x_coord):int(x_coord)+chunk.shape[1]] = target_chunk + add_chunk

pattern = re.compile(r'\((\d+), (\d+)\)')
print("Stitching Tiles...")
for f in tqdm.tqdm(glob.glob(TILE_MASK_OUTS + "\\*.tif")):
    #print(pattern.search(f))
    addAtCoordinates(FULL_MASK_CANVAS, cv2.imread(f), pattern.search(f)[1], pattern.search(f)[2])
#[replaceAtCoordinates(FULL_MASK_CANVAS, cv2.imread(f), pattern.search(f)[0], pattern.search(f)[1]) for f in glob.glob(TILE_MASK_OUTS + "\\*.tif")]

print(f"Max Value: {np.max(FULL_MASK_CANVAS)}")
#print(np.uint8((FULL_MASK_CANVAS/max(np.max(FULL_MASK_CANVAS), 1))*255))

# a[a > 3] = -101


print("Storing File!")
cv2.imwrite(IMAGE_DIR + "\\AI_MASK_" + "V" + str(VERT_SPACING) + "H" + str(HORZ_SPACING) + "_" + IN_FILE, np.uint8(FULL_MASK_CANVAS/np.max(FULL_MASK_CANVAS)*255))
print(np.max(FULL_MASK_CANVAS))

if np.max(FULL_MASK_CANVAS) > 0:
    FULL_MASK_CANVAS[FULL_MASK_CANVAS < (0.70*(np.max(FULL_MASK_CANVAS)))] = 0

cv2.imwrite(IMAGE_DIR + "\\AI_MASK_SCRUB_" + "V" + str(VERT_SPACING) + "H" + str(HORZ_SPACING) + "_" + IN_FILE, np.uint8(FULL_MASK_CANVAS/np.max(FULL_MASK_CANVAS)*255))
#print(test_dataset['cot1.tif_(1152, 1920).tif'])
#[cv2.imwrite(TILE_OUTS + "\\" + IN_FILE + f"_{i}.tif", grab_clip(CLIP_IMAGE, TILESIZE, i)) for n in test_dataset]
# TODO: MAKE THE LIST COMPREHENSION OF THE TILE CLASSIFIER WORK!!!

if False:
    for i in range(5):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis[n].astype('uint8')
        image = test_dataset[n]
        #print(image)

        scan_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(scan_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())*255

        #print((pr_mask))

        visualize(
            image=image_vis,
            predicted_mask=pr_mask
        )

# Image Clipping script

# BORROW FROM EXAMINER HERE
# Set up classifier from best model

# Run model on each image

# Store mask to file




# Stitch it back together
