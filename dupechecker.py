# dupechecker.py

import hashlib
import os
from PIL import Image

def generate_image_hash(filepath, hash_algorithm='md5'):
    """Generates a hash for an image file."""
    try:
        # Open image and convert to RGB to handle various formats consistently
        with Image.open(filepath) as img:
            img_data = img.tobytes()
        
        hasher = hashlib.new(hash_algorithm)
        hasher.update(img_data)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def compare_image_folders(folder1, folder2, hash_algorithm='md5'):
    """Compares image files in two folders using hashing."""
    hashes1 = {}
    hashes2 = {}
    
    # Generate hashes for images in folder 1
    for filename in os.listdir(folder1):
        filepath = os.path.join(folder1, filename)
        if os.path.isfile(filepath):
            img_hash = generate_image_hash(filepath, hash_algorithm)
            if img_hash:
                hashes1[img_hash] = filename

    # Generate hashes for images in folder 2
    for filename in os.listdir(folder2):
        filepath = os.path.join(folder2, filename)
        if os.path.isfile(filepath):
            img_hash = generate_image_hash(filepath, hash_algorithm)
            if img_hash:
                hashes2[img_hash] = filename

    # Compare hashes
    identical_images = []
    unique_in_folder1 = []
    unique_in_folder2 = []

    for img_hash, filename1 in hashes1.items():
        if img_hash in hashes2:
            identical_images.append((filename1, hashes2[img_hash]))
        else:
            unique_in_folder1.append(filename1)

    for img_hash, filename2 in hashes2.items():
        if img_hash not in hashes1:
            unique_in_folder2.append(filename2)

    print("\n--- Comparison Results ---")
    print(f"Identical Images (found in both folders): {len(identical_images)}")
    for f1, f2 in identical_images:
        print(f"  - {f1} (in {folder1}) and {f2} (in {folder2})")

    # print(f"\nUnique Images in {folder1}: {len(unique_in_folder1)}")
    # for filename in unique_in_folder1:
    #     print(f"  - {filename}")

    # print(f"\nUnique Images in {folder2}: {len(unique_in_folder2)}")
    # for filename in unique_in_folder2:
    #     print(f"  - {filename}")

# Example Usage:
if __name__ == "__main__":
    folder_a = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/SCD_training_data/mbn_training/classes/clustered"  # Replace with your actual folder paths
    folder_b = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/SCD_training_data/mbn_training/guesses_nojitter_06-17-2025"  # Replace with your actual folder paths
    
    # Ensure Pillow is installed: pip install Pillow
    # Ensure folders exist before running
    
    compare_image_folders(folder_a, folder_b, hash_algorithm='sha256')