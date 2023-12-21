import numpy as np

from PIL import Image



def modify_image(input_path, output_path):
    image = Image.open(input_path)
    img_array = np.array(image)

    def map_values(value):
        return (value // 16) * 16

    refixed = np.multiply(np.floor_divide(img_array, 32), 32)

    # modified_array = np.vectorize(map_values)(img_array)
    # modified_image = Image.fromarray(modified_array)
    # modified_image.save(output_path)

    modified_image = Image.fromarray(refixed)

    modified_image.save(output_path)

input_image_path = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\BASE\\cot4.tif"
output_image_path = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\BASE\\cot4_modded.png"

modify_image(input_image_path, output_image_path)