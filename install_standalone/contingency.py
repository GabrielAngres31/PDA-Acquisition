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
        '--output_folder_table',
        type = str,
        required = True,
        help = "Output file folder to save the finished histogram."
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

   
    # Label image regions
    label_image_inf = label(guess_in_image, return_num=True)
    label_image_tru = label(ground_truth_img, return_num=True)
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


