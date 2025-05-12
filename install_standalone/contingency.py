import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import skimage
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import PIL
import numpy as np

import argparse
import os
import subprocess





if False:
    import scipy.sparse as sps
    #plt.spy(cont[1:, 1:], markersize = None)
    plt.spy(IoU_cleaned, markersize = None)
    # plt.imshow(IoU[1:, 1:])
    # plt.imshow(cont[1:, 1:])
    plt.show()

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ground_truth',
        type    = str,
        help    = 'Filepath for the Ground Truth Image - MUST BE ANNOTATION'
    )
    parser.add_argument(
        '--base_image',
        type    = str,
        help    = 'Base Image to run inference on - generates inference annotation with a provided model. MUST PROVIDE A MODEL'
    )
    parser.add_argument(
        '--model_path',
        type    = str,
        help    = "Model Path for interpreting Base Image"
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
    return parser


def main(args:argparse.Namespace) -> bool:
    print("Running!")
    assert args.ground_truth, "Need to specify --ground_truth argument."
    assert os.path.exists(args.ground_truth), f"Invalid path to ground truth image. Please check --ground_truth argument.\nArgument value: {args.ground_truth}"
    assert args.guess_image or args.base_image, "Please provide either a base image with --base_image or a pregenerated inference image with --guess_image."
    if args.model_path and not(args.base_image):
        print("You've specified a model, but not an image to apply the model to. The model will not be applied.")
    if args.base_image and args.guess_image:
        print("Command cannot be run with two conflicting inputs. Please choose between providing a --base_image and --model_path argument set, and providing a --guess_image argument by itself.")
        print("Terminating operation")
        return
    
    gt = skimage.io.imread(args.ground_truth)
    if args.base_image:
        assert args.model_path, "You need to provide a model to use on the base image!"
        # print(os.path.basename(args.base_image), "CONTINGENCY.output.png")
        path_compare_image = os.path.join("inference", str(os.path.basename(args.base_image)+'CONTINGENCY.output.png'))
        halt_var = subprocess.run(f"python inference_SUF.py --model={args.model_path} --input={args.base_image} --overlap=32 --outputdir=inference --outputname=CONTINGENCY  --progress='T' --weights_only=False", shell=True, capture_output=True)
        print(path_compare_image)
        assert os.path.exists(path_compare_image), print(path_compare_image)
        compare_image = skimage.io.imread(path_compare_image)
        # cm = skimage.io.imread()
    elif args.guess_image:
        compare_image = skimage.io.imread(args.guess_image)
    

    

    # apply threshold
    # thresh_inf = threshold_otsu(compare_image)
    # bw_inf = closing(compare_image > thresh_inf, square(3))
    # thresh_tru = threshold_otsu(gt)
    # bw_tru = closing(gt > thresh_tru, square(3))

    thresh_inf = compare_image
    # compare_image = compare_image
    try: 
        compare_image.ndim == 2
    except:
        compare_image = compare_image[:,:,0]
        print(compare_image.shape)

    bw_inf = closing(compare_image > 204, square(3))
    
    
    try: 
        bw_tru = closing(gt > 204, square(3))
    except:
        gt = gt[:,:,0]
        print(gt.shape)
        bw_tru = closing(gt > 204, square(3))
    thresh_tru = threshold_otsu(gt)
    # gt = gt


    inf_out =  np.multiply(bw_inf, 255)

    # skimage.io.imsave("inference/test_skimage_out_close.png", inf_out.astype('uint8'))
    # skimage.io.imsave("inference/test_skimage_out_close.png", bw_inf.astype('uint8'))


    cleared_inf = clear_border(bw_inf)
    cleared_tru = clear_border(bw_tru)

    # label image regions
    label_image_inf = label(cleared_inf, return_num=True)
    label_image_tru = label(cleared_tru, return_num=True)
    nums =  {
            "label_image_inf":{label_image_inf[1]},
            "label_image_tru":{label_image_tru[1]},
            }

    # skimage.io.imsave("inference/test_skimage_out_cont.png", label_image_inf.astype('uint8'))
    # skimage.io.imsave("inference/test_skimage_out_cont_tru.png", label_image_tru.astype('uint8'))

    nd_inf = label_image_inf[0]
    nd_tru = label_image_tru[0]



    intersections = skimage.metrics.contingency_table(nd_tru, nd_inf)             #[N,M]
    pixelsums_annotation = intersections.sum(axis=1)    #[N,1]
    pixelsums_outputs    = intersections.sum(axis=0)    #[1,M]
    unions = pixelsums_annotation + pixelsums_outputs - intersections  #[N,M]

    IoU = intersections/unions
    #print(IoU)

    IoU_csr = IoU.tocsr()

    default_list = IoU_csr[IoU_csr > 0.001]
    # print(str(IoU_csr))

    # print(str(default_list))

    #IoU_csr[:, 0] = 0
    #IoU_csr[0, :] = 0

    #print(IoU_csr)

    bins = np.arange(0, 1, 0.05) # fixed bin size

    data = [x for x in default_list.tolist()[0]]
    print(f"Mean: {np.mean(data)}")
    print(nums)
    #plt.xlim([min(data)-0.2, max(data)+0.2])

    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('IoU data - mutant')
    plt.xlabel('IoU (bin size = 0.05)')
    plt.ylabel('Count')
    # plt.ylim()
    plt.savefig(f"inference/{os.path.basename(args.base_image)}_CONTINGENCY_testfigure.png")
    if args.show_image:
        plt.show()

    return True

if __name__ == '__main__':
    args = get_argparser().parse_args()

    ok   = main(args)
    
    if ok:
        print('Done')


