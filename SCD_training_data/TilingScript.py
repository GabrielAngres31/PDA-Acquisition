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

IMAGE_DIR = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\BASE"
TILE_OUTS = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\tiles"

IN_FILE = "cot1.tif"

IMAGE = cv2.imread(IMAGE_DIR + "\\" + IN_FILE)


TILESIZE = 64
HORZ_OVERLAP = 0
VERT_OVERLAP = 0
assert HORZ_OVERLAP < TILESIZE and VERT_OVERLAP < TILESIZE, f"Overlap values do not permit image processing! Overlap must be less than current tilesize: {TILESIZE}"


# Clip the image down
width, height = IMAGE.shape[0:2]
width_buffer = (width - TILESIZE) % (TILESIZE - HORZ_OVERLAP)
height_buffer = (height - TILESIZE) % (TILESIZE - VERT_OVERLAP)
#clip_width, clip_height = TILESIZE


CLIP_IMAGE = IMAGE[:-(width_buffer), :-(height_buffer), :]
print(CLIP_IMAGE.shape)

clip_width, clip_height = CLIP_IMAGE.shape[0:2]

# Generate all tile indices.
tile_column_indices = range(0, clip_width, TILESIZE)
tile_row_indices = range(0, clip_height, TILESIZE)

tile_corners = [(c, r) for c in tile_column_indices for r in tile_row_indices]


# Calculate the number of tiles that will be generated.
print(len(tile_corners))

# Build Tile Dataset

def grab_clip(image, tile_size, coordinates):
    x_coord, y_coord = coordinates
    return image[x_coord:x_coord+tile_size, y_coord:y_coord+tile_size, :]

if True:
    [cv2.imwrite(TILE_OUTS + "\\" + IN_FILE + f"_{i}.tif", grab_clip(CLIP_IMAGE, TILESIZE, i)) for i in tile_corners]

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

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

best_model = torch.load('C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\best_model.pth')
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

        image_gry = cv2.imread(self.images_fps[i], cv2.IMREAD_GRAYSCALE)

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
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())*255

    return pr_mask

#[cv2.imwrite(TILE_OUTS + "\\" + IN_FILE + f"_{i}.tif", grab_clip(CLIP_IMAGE, TILESIZE, i)) for n in test_dataset]
# TODO: MAKE THE LIST COMPREHENSION OF THE TILE CLASSIFIER WORK!!!

if True:
    for i in range(5):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis[n].astype('uint8')
        image = test_dataset[n]
        #print(image)

        scan_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(scan_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())*255

        print((pr_mask))

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
