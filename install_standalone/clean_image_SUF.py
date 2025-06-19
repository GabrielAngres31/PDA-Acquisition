import argparse

import skimage.io as io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.util import invert
from skimage.morphology import square, area_closing, area_opening
import numpy as np

import src.data

def main(args:argparse.Namespace) -> bool:
    """
    Processes a greyscale input image and returns a black and white image after applying filters.

    Args:
        args (argparse.Namespace): Command-line arguments containing input_path,
                                   prediction_type, filter_type, and save_image_as.

    Returns:
        bool: True if the image processing is successful.
    """
    image_in = io.imread(args.input_path) # Read in the input image
    clean_image(image_in, args.prediction_type, args.filter_type, args.save_image_as) # Apply cleaning processes to image based on prediction type, filter option, and savepath.

def clean_image(image: np.array, mode:str, filter_mode:str, saveas: str = "", return_im: bool = False): 
    """
    Cleans an input image based on specified mode and filter mode.

    Args:
        image (np.array): The input image to be cleaned.
        mode (str): Specifies the prediction type, either "outlines" or "clumps".
        filter_mode (str): Specifies the filtering method, either "otsu" or "confidence".
        saveas (str, optional): If provided, saves the cleaned image to the specified path. Defaults to None.
        return_im (bool, optional): If True, returns the cleaned image. Defaults to False.
    """
    if len(image.shape) == 3:
        image = rgb2gray(image)  # Convert color image to grayscale if it has 3 channels
    print(image.shape)  # Print the shape of the image

    if filter_mode == "otsu":
        # Apply Otsu's thresholding for filtering
        if mode == "outlines":
            # Process for "outlines" mode
            otsu_trsh_num = threshold_otsu(image)  # Calculate Otsu's threshold
            # Perform area closing to fill dark patches
            otsu_fill = area_closing(image > otsu_trsh_num, connectivity=square(3), area_threshold=2500)
            otsu_invt = invert(image > otsu_trsh_num)  # Invert the thresholded image

            inners = np.logical_and(otsu_fill, otsu_invt)  # Find the intersection of filled and inverted images
            otsu_clr = area_opening(inners, area_threshold=200)  # Perform area opening to remove small objects
            final_image = otsu_clr
        elif mode == "clumps":
            # Process for "clumps" mode
            print("Thresholding...")
            thresh_value = threshold_otsu(image)  # Calculate Otsu's threshold
            thresh_image = (image > thresh_value)  # Apply thresholding
            image_numpy = np.asarray(thresh_image)  # Convert to numpy array

            print("Closing...")
            image_closing = area_closing(image_numpy, area_threshold=0)  # Perform area closing
            print("Opening...")
            image_opening = area_opening(image_closing, area_threshold=100)  # Perform area opening
            final_image = image_opening
    elif filter_mode == "confidence":
        # Apply confidence-based filtering
        final_image = image / np.max(image) > 0.58  # Normalize image and apply a confidence threshold
        final_image = area_closing(final_image, area_threshold=750)  # Perform area closing
        final_image = area_opening(final_image, area_threshold=350)  # Perform area opening

    if saveas:
        # Save the cleaned image if saveas path is provided
        src.data.save_image(f"cleaned_images_default/{saveas}.png", final_image.astype(np.float32))
    if return_im:
        return final_image  # Return the cleaned image if return_im is True

def get_argparser() -> argparse.ArgumentParser:
    """
    Creates and configures an ArgumentParser for command-line arguments.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Image to find chunks for.'
    )
    parser.add_argument(
        '--prediction_type',
        type=str,
        default='clumps',
        choices=['clumps', 'outlines'],
        help='Whether the mask being read is an outline of the stomata or the stomata themselves.'
    )
    parser.add_argument(
        '--filter_type',
        type=str,
        default='otsu',
        choices=['confidence', 'otsu'],
        help='Whether to filter on absolute pixel brightness or an otsu threshold.'
    )
    parser.add_argument(
        '--save_image_as',
        type=str,
        default=None,
        help='Optional - saves an image of the cleaned image to a directory/name.jpg you specify'
    )
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()  # Parse command-line arguments
    ok = main(args)  # Call the main function
    if ok:
        print('Done')  # Print "Done" if the main function completes successfully