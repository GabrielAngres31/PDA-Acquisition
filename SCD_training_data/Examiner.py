import torch
from torchvision import transforms
from PIL import Image
import numpy  as np
import segmentation_models_pytorch as smp
import cv2
import sys
import os
import albumentations as albu
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset
import inspect


print("Finished Imports!")

if __name__ == '__main__':

    def to_tensor(x, **kwargs):
        return x.transpose(2, 0, 1).astype('float32')
    
    DATA_DIR = 'C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\full_training_set'


    #scan_dir = os.path.join(DATA_DIR, 'train')

    scan_dir = os.path.join(DATA_DIR, 'test')


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



    # Lets look at data we have

    def get_validation_augmentation():
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            albu.PadIfNeeded(64, 64)
        ]
        return albu.Compose(test_transform)

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

    for i in range(5):
        n = np.random.choice(len(test_dataset))

        image_vis = test_dataset_vis[n].astype('uint8')
        image = test_dataset[n]
        print(image)

        scan_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(scan_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        visualize(
            image=image_vis,
            predicted_mask=pr_mask
        )