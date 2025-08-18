# imagecleaner.py

import argparse
import os
import sys
import warnings

import numpy         as np
from skimage.filters import threshold_otsu
from skimage.io      import imread
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, area_opening, area_closing, binary_erosion, binary_dilation
from skimage.segmentation import clear_border
from PIL import Image

import tqdm


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
        action='store_true',
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
        "--remove_sticks",
        default = False,
        action='store_true',
        help = "Set this to True to remove stick-like projections from rounded features."
    )
    parser.add_argument(
        "--savepath",
        type = str,
        help = "Filepath to save generated image to."
    )
    return parser


def main(args:argparse.Namespace) -> bool:

    guess_in_image = imread(args.image_in)

    if guess_in_image.ndim == 3:
        guess_in_image = guess_in_image[:,:,0]

    def process_image_largeobjects(img, threshold=188, cutoff=3750):
        """
        Loads an image, thresholds it to binary, and removes objects that have a greater concave or rectangular area than convex area.

        Args:
            image_filepath (str): The path to the input image.
            threshold_constant (int): The threshold value (0-255). Pixels above this
                                    become 255, others become 0. Default set to hand-tuned value of 10.
            min_object_size_cutoff (int): The maximum size (in pixels) for an object
                                          to be KEPT. Objects *larger* than this
                                          will be removed (set to black).
                                          Default set to hand-tuned value of 3750.
        Returns:
            numpy.ndarray: The processed image.
        """

        try:
            binary_image = (img > threshold).astype(np.uint8)
            labels = label(binary_image, connectivity = 2)

            large_objects_mask = np.zeros_like(binary_image,dtype=bool)

            for region in tqdm.tqdm(regionprops(labels)):
                if region["area_bbox"]/region["area"] > 3:
                    for r, c in region.coords:
                        large_objects_mask[r,c] = True
            img_out = img.copy()
            img_out[large_objects_mask] = 0
            return img_out

        except FileNotFoundError:
            print(f"Error: Image file not found at {img}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    condition_gi = None

    if args.mode == None:
        bw_inf = guess_in_image
    elif args.mode == "flat":
        condition_gi = guess_in_image > 204
    elif args.mode == "full_filter":
        guess_in_image = process_image_largeobjects(guess_in_image)
        thresh_inf = threshold_otsu(guess_in_image)
        condition_gi = guess_in_image > thresh_inf
    elif args.mode == "otsu":
        thresh_inf = threshold_otsu(guess_in_image)
        condition_gi = guess_in_image > thresh_inf
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    if condition_gi is not None:
        bw_inf = closing(condition_gi, square(3))   

    # ---
    # New section for removing stick-like projections (moved to here)
    if args.remove_sticks:
        selem = square(4)
        bw_inf = binary_erosion(bw_inf, selem)
        bw_inf = binary_dilation(bw_inf, selem)
        bw_inf = area_opening(bw_inf, area_threshold=50)
    # ---

    rmv_sml = args.remove_small_thresh
    if rmv_sml > 0:
        bw_inf = area_opening(bw_inf, area_threshold=rmv_sml)    

    bw_inf = area_closing(bw_inf, area_threshold=3000) 

    cleared_inf = clear_border(bw_inf)

    final_image = Image.fromarray(cleared_inf)

    if args.show_image:
        final_image.show()
    if args.savepath:
        save_dir = os.path.dirname(args.savepath)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        try:
            final_image.save(args.savepath)
        except Exception as e:
            print(f"Error saving image to {args.savepath}: {e}")

    return True

if __name__ == '__main__':
    args = get_argparser().parse_args()

    ok   = main(args)
    
    if ok:
        pass
        print('Done')