import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Define the paths to the image and annotation folders
img_folder = "all"
annot_folder = "allannot"

# Function to load and preprocess images
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [64, 64])
    img = tf.image.convert_image_dtype(img, tf.float32)  # Normalize to [0, 1]
    return img

# Function to load and preprocess annotations
def load_annotation(annot_path):
    annot = tf.io.read_file(annot_path)
    annot = tf.image.decode_png(annot, channels=1)
    annot = tf.image.resize(annot, [64, 64])
    annot = tf.cast(annot > 128, tf.uint8)  # Convert to 1-bit mask
    return annot

# Create a list of file paths for images and annotations
img_paths = [os.path.join(img_folder, filename) for filename in os.listdir(img_folder) if filename.endswith(".png")]
annot_paths = [os.path.join(annot_folder, filename) for filename in os.listdir(annot_folder) if filename.endswith(".png")]

# Split the dataset into training and testing sets
img_train_paths, img_test_paths, annot_train_paths, annot_test_paths = train_test_split(
    img_paths, annot_paths, test_size=0.2, random_state=42
)

# Create TensorFlow datasets for training and testing
train_dataset = tf.data.Dataset.from_tensor_slices((img_train_paths, annot_train_paths))
train_dataset = train_dataset.map(lambda img, annot: (load_image(img), load_annotation(annot)))

test_dataset = tf.data.Dataset.from_tensor_slices((img_test_paths, annot_test_paths))
test_dataset = test_dataset.map(lambda img, annot: (load_image(img), load_annotation(annot)))

# Batch and prefetch the datasets for better performance
batch_size = 32
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Define a simple segmentation model (you can replace this with your own model)
def create_model():
    print("1. All OK!")
    model = models.Sequential()
    print("2. All OK!")
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    print("3. All OK!")
    model.add(layers.MaxPooling2D((2, 2)))
    print("4. All OK!")
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    print("5. All OK!")
    model.add(layers.MaxPooling2D((2, 2)))
    print("6. All OK!")
    model.add(layers.Flatten())
    print("7. All OK!")
    model.add(layers.Dense(64, activation='relu'))
    print("8. All OK!")
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary segmentation
    print("9. All OK!")
    return model

# Create the segmentation model
model = create_model()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using the training dataset
model.fit(train_dataset, epochs=5, validation_data=test_dataset)