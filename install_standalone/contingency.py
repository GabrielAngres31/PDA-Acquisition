import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy             as np
from scipy.sparse import SparseEfficiencyWarning
from skimage.filters      import threshold_otsu
from skimage.io           import imread
from skimage.measure      import label, regionprops
from skimage.metrics      import contingency_table
from skimage.morphology   import closing, square
from skimage.segmentation import clear_border

warnings.simplefilter('ignore', SparseEfficiencyWarning) # Suppress SparseEfficiencyWarning

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ground_truth',
        type    = str,
        help    = 'Filepath for the Ground Truth Image - MUST BE ANNOTATION'
    )
    parser.add_argument(
        '--guess_image',
        type    = str,
        help    = 'Annotation generated from machine inference. MUST CORRESPOND TO GT IMAGE'
    )
    parser.add_argument(
        '--show_image',
        default = False,
        help    = "Set this to True to see the histogram of residual errors. Not recommended for batch operations."
    )
    parser.add_argument(
        '--texttag',
        type = str,
        default = "",
        help    = "Label for output histogram. Default none."
    )
    parser.add_argument(
        '--mode',
        type=str,
        required = True,
        choices = ("flat", "otsu", "full_filter"),
        help = "Which mode to use. Either uses a flat threshold of any pixel more than 204 on [0, 255] or an otsu threshold."
    )
    parser.add_argument(
        '--output_folder_table',
        type = str,
        required = True,
        help = "Output file folder to save the finished table."
    )
    return parser


def main(args:argparse.Namespace) -> bool:
    # print("Running!")
    assert args.ground_truth, "Need to specify --ground_truth argument."
    assert args.guess_image,  "Need to specify --guess_image argument."
    assert os.path.exists(args.ground_truth), f"Invalid path to ground truth image. Please check --ground_truth argument.\nArgument value: {args.ground_truth}"
    assert os.path.exists(args.guess_image),  f"Invalid path to guess image. Please check --guess_image argument.\nArgument value: {args.guess_image}"
    
    ground_truth_img = imread(args.ground_truth) # Read in the ground truth image. Assumed to be black and white. HOWEVER, can be substituted with a greyscale image if comparing two inferences.
    guess_in_image =   imread(args.guess_image)  # Read in the guess image. Assumed to be either black and white, or greyscale on [0,255]

    if guess_in_image.ndim == 3:
        guess_in_image = guess_in_image[:,:,0]     # Ensure that both images have only two dimensions, as RGB values should not be present in the image.
    if ground_truth_img.ndim == 3:
        ground_truth_img = ground_truth_img[:,:,0] # --    --    --    --    --    --    --    --    --    --    --    --    --    --    --    --    --

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
    condition_gt = None

    if args.mode == None:
        bw_inf = guess_in_image   # Make copies of the input images to modify.
        bw_tru = ground_truth_img # --    --    --    --    --    --    --  --
    elif args.mode == "flat":
        # Threshold both images on a hand-tuned value on [0,255]
        condition_gt = ground_truth_img > 204
        condition_gi = guess_in_image   > 204
    elif args.mode == "full_filter":
        # Apply large clumps filter, then threshold on Otsu.
        guess_in_image = process_image_largeobjects(guess_in_image)
        thresh_inf = threshold_otsu(guess_in_image)
        thresh_tru = threshold_otsu(ground_truth_img)
        condition_gt = ground_truth_img > thresh_tru
        condition_gi = guess_in_image > thresh_inf
    elif args.mode == "otsu":
        # Threshold both images on an Otsu threshold.
        thresh_inf = threshold_otsu(guess_in_image)
        thresh_tru = threshold_otsu(ground_truth_img)
        condition_gt = ground_truth_img > thresh_tru
        condition_gi = guess_in_image > thresh_inf
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    if condition_gi is not None and condition_gt is not None:
        bw_inf = closing(condition_gi, square(3))        
        bw_tru = closing(condition_gt, square(3))

    cleared_inf = clear_border(bw_inf)
    cleared_tru = clear_border(bw_tru)

    # Label image regions
    label_image_inf = label(cleared_inf, return_num=True)
    label_image_tru = label(cleared_tru, return_num=True)
    # nums =  {
    #         "label_image_inf":{label_image_inf[1]},
    #         "label_image_tru":{label_image_tru[1]},
    #         }
    
    nd_inf = label_image_inf[0]
    nd_tru = label_image_tru[0]

    try:
        intersections = contingency_table(nd_tru, nd_inf)             #[N,M]
    except:
        if nd_inf.shape != nd_tru.shape: # Diagnostic for image mismatch.
            print(f"INF: {nd_inf.shape}")
            print(args.guess_image)
            print(f"TRU: {nd_tru.shape}")
            print(args.ground_truth)
            sys.exit()
    pixelsums_annotation = intersections.sum(axis=1)                   #[N,1]
    pixelsums_outputs    = intersections.sum(axis=0)                   #[1,M]
    unions = pixelsums_annotation + pixelsums_outputs - intersections  #[N,M]

    IoU = intersections/unions # Generate IoU

    IoU_csr = IoU.tocsr()

    default_list = IoU_csr[IoU_csr > 0.001]

    IoU_csr[:, 0] = 0
    IoU_csr[0, :] = 0

    bins = np.arange(0, 1, 0.05) # Fixed bin size

    data = [x for x in default_list.tolist()[0]]
    print(args.ground_truth)
    print(f"Mean: {np.mean(data)}")
    # print(nums)
    #plt.xlim([min(data)-0.2, max(data)+0.2])

    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('IoU data - mutant')
    plt.xlabel('IoU (bin size = 0.05)')
    plt.ylabel('Count')
    if args.show_image:
        plt.show()
    plt.savefig(f"inference/contingencycompare/{os.path.basename(args.guess_image)}_{args.texttag}.png") #TODO: save folder as argument


    return True

if __name__ == '__main__':
    args = get_argparser().parse_args()

    ok   = main(args)
    
    if ok:
        pass
        print('Done')


