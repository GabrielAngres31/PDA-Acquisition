# imagecleaner.py

import argparse
import os
import sys
import warnings

import numpy             as np
from skimage.filters      import threshold_otsu
from skimage.io           import imread
from skimage.measure      import label, regionprops
from skimage.morphology   import closing, square, area_opening
from skimage.segmentation import clear_border
from PIL import Image

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image_in',
        type    = str,
        help    = 'Image to be cleaned'
    )
    parser.add_argument(
        '--show_image',
        default = False,
        help    = "Set this to True to see the cleaned image. Not recommended for batch operations."
    )
    parser.add_argument(
        '--mode',
        type=str,
        required = True,
        choices = ("flat", "otsu", "full_filter"),
        help = "Which mode to use. Either uses a flat threshold of any pixel more than 204 on [0, 255] or an otsu threshold."
    )
    parser.add_argument(
        "--remove_small_thresh",
        type = int,
        default = 250,
        help = "Threshold below which all smaller clumps are removed."
    )
    parser.add_argument(
        "--savepath",
        type = str,
        help = "Filepath to save generated image to."
    )
    return parser


def main(args:argparse.Namespace) -> bool:

    guess_in_image = imread(args.image_in)  # Read in the guess image. Assumed to be either black and white, or greyscale on [0,255]

    if guess_in_image.ndim == 3:
        guess_in_image = guess_in_image[:,:,0] # --    --    --    --    --    --    --    --    --    --    --    --    --    --    --    --    --

    def process_image_largeobjects(img, threshold=10, cutoff=2500):
        """
        Loads an image, thresholds it to binary, and removes objects larger than a cutoff.

        Args:
            image_filepath (str): The path to the input image.
            threshold_constant (int): The threshold value (0-255). Pixels above this
                                    become 255, others become 0. Default set to hand-tuned value of 10.
            min_object_size_cutoff (int): The maximum size (in pixels) for an object
                                        to be KEPT. Objects *larger* than this
                                        will be removed (set to black).
                                        Default set to hand-tuned value of 2500.
        Returns:
            numpy.ndarray: The processed image.
        """

        try:
            binary_image = (img > threshold).astype(np.uint8) # Generate binary mask of pixels above the threshold
            labels = label(binary_image, connectivity = 2) # Find clumps

            large_objects_mask = np.zeros_like(binary_image,dtype=bool) # Generate an array of zeros the same size as binary_image

            for region in regionprops(labels):
                # If the object's area is greater than the min_object_size_cutoff,
                # it's considered a "large object" that should be removed.
                if region.area > cutoff: # Set the pixels corresponding to this large object in the mask to True
                    for r, c in region.coords:
                        large_objects_mask[r,c] = True
            img_out = img.copy() # Generate copy of img to eliminate large clumps from
            img_out[large_objects_mask] = 0 # Eliminate large clumps
            return img_out

        except FileNotFoundError:
            print(f"Error: Image file not found at {img}") # If the file is not found, notify the user
            return None
        except Exception as e: # If a different exception occurs, notify the user
            print(f"An error occurred: {e}")
            return None
    
    condition_gi = None

    if args.mode == None:
        bw_inf = guess_in_image   # Make copies of the input images to modify.
    elif args.mode == "flat":
        # Threshold both images on a hand-tuned value on [0,255]
        condition_gi = guess_in_image   > 204
    elif args.mode == "full_filter":
        # Apply large clumps filter, then threshold on Otsu.
        guess_in_image = process_image_largeobjects(guess_in_image)
        thresh_inf = threshold_otsu(guess_in_image)
        condition_gi = guess_in_image > thresh_inf
    elif args.mode == "otsu":
        # Threshold both images on an Otsu threshold.
        thresh_inf = threshold_otsu(guess_in_image)
        condition_gi = guess_in_image > thresh_inf
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    if condition_gi is not None:
        bw_inf = closing(condition_gi, square(3))  

    rmv_sml = args.remove_small_thresh
    if rmv_sml > 0:
        bw_inf = area_opening(bw_inf, area_threshold=rmv_sml)      

    cleared_inf = clear_border(bw_inf)

    final_image = Image.fromarray(cleared_inf)

    if args.show_image:
        final_image.show()
    if args.savepath:
        final_image.save(args.savepath)

    return True

if __name__ == '__main__':
    args = get_argparser().parse_args()

    ok   = main(args)
    
    if ok:
        pass
        print('Done')

