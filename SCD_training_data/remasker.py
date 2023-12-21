from PIL import Image
import glob
import os
import tqdm

def process_image(file_path):
    with Image.open(file_path) as img:

        img = img.convert('1')
        
        img_data = img.getdata()
        #print(img_data)
        #print([pixel for pixel in img_data])
        #### modified_data = [(0 if pixel == 255 else 1 if pixel == 0 else pixel) for pixel in img_data]
        #print([pixel for pixel in modified_data])
        modified_data = [(1 if pixel == 255 else pixel) for pixel in img_data]
        
        modified_img = Image.new(img.mode, img.size)
        modified_img.putdata(modified_data)

        
        modified_file_path = file_path.replace(".png", ".png")#, "_modified.png")
        modified_img.save(modified_file_path)
        #print(f"Processed: {file_path} -> {modified_file_path}")

def process_images_in_folder(folder_path):
    
    pattern = os.path.join(folder_path, '**/*.png')
    png_files = glob.glob(pattern, recursive=True)

    for png_file in tqdm.tqdm(png_files):
        process_image(png_file)

if __name__ == "__main__":
    #target_folder = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\generated\\test\\anno"
    #target_folder = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\example_training_set\\"
    target_folder = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\Train_stomata\\old_folder_storage"
    folders = ["testannot", "trainannot", "valannot"]
    [process_images_in_folder(os.path.join(target_folder, i)) for i in folders]