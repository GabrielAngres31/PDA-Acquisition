import keras
import tensorflow as tf
import numpy as np


from tifffile import imread, imwrite

#tiff_in = imread("C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\Generated\\sourcetiff.tiff")

def RGB_to_flat_vec(tiff_file):
    return np.asarray([item for item in [[pixel[0] for pixel in line] for line in tiff_file]]).flatten()

#print(len(RGB_to_flat_vec(tiff_in)))

image_dataset = tf.keras.utils.image_dataset_from_directory("C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\Generated", 
                                            labels = None,
                                            color_mode="grayscale")

# USEFUL FOR TESTING LATER
image = tf.keras.utils.load_img("C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\Generated\\sourcetiff.png")
input_arr = tf.keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
#predictions = model.predict(input_arr)


