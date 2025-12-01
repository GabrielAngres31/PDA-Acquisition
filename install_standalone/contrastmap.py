# contrast_map.py

import numpy as np
from PIL import Image

import argparse




def RGBgen_from_monodiff_vector(arr, threshold):
    
    # Calculates the R value corresponding to the monochrome cell, to make the first image ORANGE (in combination with the G value)
    R = np.maximum( arr, 0)
    # Calculates the B value corresponding to the monochrome cell, to make the first image BLUE (in combination with the G value)
    B = np.maximum(-arr, 0)

    # Threshold is used to artificially inflate minor differences so that they can be seen in the first place.
    R[np.logical_and(arr > 0, arr <  threshold)] = threshold
    B[np.logical_and(arr < 0, arr > -threshold)] = threshold

    # GREEN value that scales with half the magnitude of the difference in the monochrome values.
    G = np.abs(arr)//2

    # if num == 0:
    #     R, G, B = 0
    # RGB array returned for saving and display with PIL.Image
    return np.stack([R,G,B], axis=-1).astype(np.uint8)

def RGBgen_from_vecdiff(arr0, arr1, threshold):
    magenta_mask = (arr0 == arr1) & (arr0 != 0)
    print(np.where(arr0 == arr1))
    arr = np.subtract(arr0,arr1)

    # Calculates the R value corresponding to the monochrome cell, to make the first image ORANGE (in combination with the G value)
    R = np.maximum( arr, 0)
    # Calculates the B value corresponding to the monochrome cell, to make the first image BLUE (in combination with the G value)
    B = np.maximum(-arr, 0)

    R[np.logical_and(arr > 0, arr <  threshold)] = threshold
    B[np.logical_and(arr < 0, arr > -threshold)] = threshold

    G = np.abs(arr)//2
    
    # R = np.where(magenta_mask, 255, R)
    # G = np.where(magenta_mask,   0, G)
    # B = np.where(magenta_mask, 255, B)
    R = np.where(magenta_mask == 1, 255, np.maximum( arr,0))
    G = np.where(magenta_mask == 1,   0, np.abs(arr)//2)
    B = np.where(magenta_mask == 1, 255, np.maximum(-arr,0))
    # G[magenta_mask] = 0
    # B[magenta_mask] = 255

    # R = np.where(magenta_mask, 255, R)
    # B = np.where(magenta_mask, 255, B)
    # G = np.where(magenta_mask,   0, G)

    # R = np.clip(R, 0, 255)
    # B = np.clip(B, 0, 255)
    # G = np.clip(G, 0, 255)

    return np.stack([R,G,B], axis=-1).astype(np.uint8)



def generate_compare(former_img, latter_img, threshold, concordance_highlight):

    # Generates an array of numbers from -255 to 255 inclusive.
    # A positive value represents a feature MORE present in the first image than the second. This will be shown in ORANGE.
    # A negative value represents a feature LESS present in the first image than in the second. This will be shown in BLUE.
    print(concordance_highlight)
    if concordance_highlight:
        delta_RGB_diff = RGBgen_from_vecdiff(former_img.astype(np.int16), latter_img.astype(np.int16), threshold)
        return delta_RGB_diff
    else:    
        delta = np.subtract(former_img.astype(np.int16), latter_img.astype(np.int16))
        # Vectorized apply to convert [-255, 255] psuedo-monochrome to ([0, 255], [0, 255], [0, 255]) RGB.
        delta_RGB_mono = RGBgen_from_monodiff_vector(delta, threshold)
        return delta_RGB_mono


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--base_img',
        type     = str,
        required = True,
        help     = 'Path to base image.'
    )
    parser.add_argument(
        '--compare_img',
        type     = str,
        required = True,
        help     = 'Path to comparison image.'
    )
    parser.add_argument(
        '--threshold',
        type     = int,
        choices  = range(256),
        default  = 100,
        metavar  = "[0-255]",
        help     = 'Absolute value threshold below which everything is given the shaded threshold value. Set to 0 to show all absolute differences. Set to 255 to highlight all pixels that do not exactly match, at maximum brightness. Set to an intermediate value to arbitratily visualize small differences.'
    )
    parser.add_argument(
        '--output_path',
        type     = str,
        help     = 'Output file to save image.'
    )
    parser.add_argument(
        '--show_image',
        type     = bool,
        help     = 'Whether to display the image using Image imshow().'
    )
    parser.add_argument(
        '--concordance_highlight',
        type     = bool,
        default  = False,
        help     = 'Whether to recolor areas of perfect concordance as pink.'
    )
    return parser


def main(args:argparse.Namespace) -> bool:
    print("RUNNING!")
    if not args.output_path and not args.show_image:
        raise Exception("You are neither saving nor displaying an image. Please provide a savepath to --output_path, or set --show_image to True.")
    base    = np.asarray(Image.open(args.base_img)) 
    compare = np.asarray(Image.open(args.compare_img))
    img_out = Image.fromarray(generate_compare(base, compare, args.threshold, args.concordance_highlight))
    if args.output_path:
        img_out.save(args.output_path+".png")
    if args.show_image:
        img_out.show()
    
    return True

if __name__ == '__main__':
    args = get_argparser().parse_args()

    ok   = main(args)
    
    if ok:
        pass