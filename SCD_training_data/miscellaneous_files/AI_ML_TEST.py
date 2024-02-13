import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import glob

# Define the paths to the image and annotation folders
# img_folder = "img"
# annot_folder = "img_annot"

img_folder = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\generated\\test\\base"
annot_folder = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\generated\\test\\anno"

# Function to load and preprocess images
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [64, 64])
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize to [0, 1]
    img = tf.expand_dims(img, axis = -1)
    return img

# Function to load and preprocess annotations
def load_annotation(annot_path):
    annot = tf.io.read_file(annot_path)
    annot = tf.image.decode_png(annot, channels=1)
    annot = tf.image.resize(annot, [64, 64])
    annot = tf.cast(annot > 128, tf.uint8)  # Convert to 1-bit mask
    annot = tf.expand_dims(annot, axis = -1)
    return annot

# Create a list of file paths for images and annotations
# img_paths = [os.path.join(img_folder, filename) for filename in glob.glob(img_folder + "\\**\\*.png", recursive = True)]
# annot_paths = [os.path.join(annot_folder, filename) for filename in glob.glob(annot_folder)]

img_paths = [filename for filename in glob.glob(img_folder + "\\**\\*.png", recursive = True)]
annot_paths = [filename for filename in glob.glob(annot_folder + "\\**\\*.png", recursive = True)]

# Split the dataset into training and testing sets
img_train_paths, img_test_paths, annot_train_paths, annot_test_paths = train_test_split(
    img_paths, annot_paths, test_size=0.2, random_state=42
)

print("DATASET CREATION")
# Create TensorFlow datasets for training and testing
train_dataset = tf.data.Dataset.from_tensor_slices((img_train_paths, annot_train_paths))
train_dataset = train_dataset.map(lambda img, annot: (load_image(img), load_annotation(annot)))

test_dataset = tf.data.Dataset.from_tensor_slices((img_test_paths, annot_test_paths))
test_dataset = test_dataset.map(lambda img, annot: (load_image(img), load_annotation(annot)))

print("BATCHING")
# Batch and prefetch the datasets for better performance
batch_size = 32
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("MODEL CREATION")
# Define a simple segmentation model (you can replace this with your own model)
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    #######model.add(layers.Conv2D(1, (1, 1), activation = 'sigmoid'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary segmentation
    return model

print("SEG")
# Create the segmentation model
model = create_model()

# Compile the model
print("COMPILE")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using the training dataset
print("DATASET CREATION")
model.fit(train_dataset, epochs=5, validation_data=test_dataset)