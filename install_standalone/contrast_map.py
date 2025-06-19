# contrast_map.py

import numpy as np
from PIL import Image

import argparse




def RGBgen_from_monodiff_vector(num, threshold):
    
    # Calculates the R value corresponding to the monochrome cell, to make the first image ORANGE (in combination with the G value)
    R = np.maximum( num, 0)
    # Calculates the B value corresponding to the monochrome cell, to make the first image BLUE (in combination with the G value)
    B = np.maximum(-num, 0)

    # THreshold is used to artificially inflate minor differences so that they can be seen in the first place.
    R[np.logical_and(num > 0, num <  threshold)] = threshold
    B[np.logical_and(num < 0, num > -threshold)] = threshold

    # GREEN value that scales with half the magnitude of the difference in the monochrome values.
    G = np.abs(num)//2

    # RGB array returned for saving and display with PIL.Image
    return np.stack([R,G,B], axis=-1).astype(np.uint8)


def generate_compare(former_img, latter_img, threshold):
    # Generates an array of numbers from -255 to 255 inclusive.
    # A positive value represents a feature MORE present in the first image than the second. This will be shown in ORANGE.
    # A negative value represents a feature LESS present in the first image than in the second. This will be shown in BLUE.
    delta = np.subtract(former_img.astype(np.int16), latter_img.astype(np.int16))
    
    # Vectorized apply to convert [-255, 255] psuedo-monochrome to ([0, 255], [0, 255], [0, 255]) RGB.
    delta_RGB = RGBgen_from_monodiff_vector(delta, threshold)
    
    return delta_RGB


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
    return parser


def main(args:argparse.Namespace) -> bool:
    if not args.output_path and not args.show_image:
        raise Exception("You are neither saving nor displaying an image. Please provide a savepath to --output_path, or set --show_image to True.")
    base    = np.asarray(Image.open(args.base_img)) 
    compare = np.asarray(Image.open(args.compare_img))
    img_out = Image.fromarray(generate_compare(base, compare, args.threshold))
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






