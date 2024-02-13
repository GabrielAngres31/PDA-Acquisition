import keras
import tensorflow as tf
import numpy as np
import os
import sys
from tifffile import imread, imwrite
from PIL import Image as im

DIR_CWD = os.getcwd()
DIR_SOURCE = os.path.join(DIR_CWD, "source_images")
DIR_IMAGESET = os.path.join(DIR_SOURCE, "generated")
DIR_ABSENT = os.path.join(DIR_IMAGESET, "ABSENT") 
DIR_PARTIAL= os.path.join(DIR_IMAGESET, "PARTIAL")
DIR_WHOLE  = os.path.join(DIR_IMAGESET, "WHOLE")

#np.set_printoptions(threshold=sys.maxsize)


from tifffile import imread, imwrite

#tiff_in = imread("C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\Generated\\sourcetiff.tiff")

def RGB_to_flat_vec(tiff_file):
    return np.asarray([item for item in [[pixel[0] for pixel in line] for line in tiff_file]]).flatten()

#print(len(RGB_to_flat_vec(tiff_in)))

#image_dataset = tf.keras.utils.image_dataset_from_directory(DIR_IMAGESET, 
#                                            color_mode="grayscale",
#                                            validation_split=0.4,
#                                            subset = "both",
#                                            seed = 3735928559,
#                                            label_mode = "categorical",
#                                            image_size = (64, 64),
#                                            labels = 'inferred')


model = tf.keras.models.Sequential()
max_pool_2d_64_to_32 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")
max_pool_2d_32_to_16 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid')

model.add(max_pool_2d_64_to_32)
model.add(max_pool_2d_32_to_16)

# 2x Max Pooling layers to reduce input from 64x64 to 16x16


# USEFUL FOR TESTING LATER
image = tf.keras.utils.load_img(os.path.join(DIR_IMAGESET, "COT1_912x_1736y.png"))
input_arr = tf.keras.utils.img_to_array(image)

input_arr = np.array([x for x in input_arr[:,:,0]]).astype(int)  # Convert single image to a batch.

whole_in = tf.keras.utils.img_to_array(tf.keras.utils.load_img(os.path.join(DIR_SOURCE, "BASE\\cot1_TESTSEG.tiff")))[:,:,0]
#max_pool_2d_64_to_32(input_arr)
#predictions = model.predict(input_arr)

#print(input_arr)

def byhandMaxPool(array_in):
    
    assert array_in.shape[0] % 2 == 0
    assert array_in.shape[1] % 2 == 0

    iterativeResult = np.zeros((array_in.shape[0]//2, array_in.shape[1]//2))
    for i in list(range(0, array_in.size, 4)):
        
        y = i//(array_in.shape[0]*2)*2
        x = i%(array_in.shape[1]//2)

        print(f"{i}:_{x}_{y}_")
        try:
            iterativeResult[y//2][x//2] = np.max(array_in[y:y+2, x:x+2])
        except ValueError:
            pass
    return iterativeResult.astype(int)
    #M, N = input_arr.shape
    #maxPooled = input_arr[:M, :N].reshape(M//2, N//2, 2).max(axis=(1,3))

    #return maxPooled

def maxpool_manual(array_in):
    
    assert array_in.shape[0] % 2 == 0
    assert array_in.shape[1] % 2 == 0

    iterativeResult = np.zeros((array_in.shape[0]//2, array_in.shape[1]//2))
    for x in range(0, array_in.shape[1], 2):
        for y in range(0, array_in.shape[0], 2):
            iterativeResult[y//2][x//2] = np.max(array_in[y:y+2, x:x+2])
            #print(f"_{x}_{y}_")

    return iterativeResult.astype(int)

def avgpool_manual(array_in):
    assert array_in.shape[0] % 2 == 0
    assert array_in.shape[1] % 2 == 0

    iterativeResult = np.zeros((array_in.shape[0]//2, array_in.shape[1]//2))
    for x in range(0, array_in.shape[1], 2):
        for y in range(0, array_in.shape[0], 2):
            iterativeResult[y//2][x//2] = np.mean(array_in[y:y+2, x:x+2])
            #print(f"_{x}_{y}_")

    return iterativeResult.astype(int)

def freqmap_manual(array_in):
    array_out = np.zeros(shape = (array_in.shape[0], array_in.shape[1]))
    unique, counts = np.unique(array_in, return_counts = True)
    counts = np.asarray([min(sum(counts)//125, x) for x in counts])
    print(counts[np.argsort(counts)][-5:])
    counts = (((1-counts/np.max(counts))*255)).astype(int)

    re_map = dict(zip(unique, counts))


    for y in range(array_in.shape[0]):
        for x in range(array_in.shape[1]):
            array_out[y][x] = re_map[array_in[y][x]]
    print(array_out)
    return array_out

def freqmap_manual_log(array_in, division = 75):
    array_out = np.zeros(shape = (array_in.shape[0], array_in.shape[1]))
    unique, counts = np.unique(array_in, return_counts = True)
    counts = np.asarray([np.log(min(sum(counts)//division, x)) for x in counts])
    print(counts[np.argsort(counts)][-5:])
    counts = (((1-counts/np.max(counts))*255)).astype(int)

    re_map = dict(zip(unique, counts))


    for y in range(array_in.shape[0]):
        for x in range(array_in.shape[1]):
            array_out[y][x] = re_map[array_in[y][x]]
    print(array_out)
    return array_out

def selective_edge_detection(array_in, lower, upper, func):
    
    assert lower in list(range(256))
    assert upper in list(range(256))
    assert upper > lower

    array_out = np.zeros(shape = (array_in.shape[0], array_in.shape[1]))

    height, width = array_in.shape

    def inBounds(x_ind, y_ind):
        return 0<=x_ind<width and 0<=y_ind<height
    
    def getNeighbors(arr, x_ind, y_ind):
        result = []
        pairs = zip([-1, -1, -1,  0, 0,  1, 1, 1],
                    [-1,  0,  1, -1, 1, -1, 0, 1])
        for xc, yc in pairs:
            if inBounds(x_ind+xc, y_ind+yc):
                
                result.append(arr[y_ind+yc][x_ind+xc]) 
        return result
    
    piskels = 0

    for y in range(height):
        for x in range(width):
            
            #print(array_in.shape)
            
            delta = func([abs(n - array_in[y][x]) for n in getNeighbors(array_in, x, y)])
            
            if delta > lower and delta < upper:
                array_out[y][x] = 255
            else:
                array_out[y][x] = 0
    
    return array_out



#select_test_var = selective_edge_detection(whole_in, 20, 180)

#select_test_img = im.fromarray(select_test_var)
#select_test_img.show()

im.fromarray(selective_edge_detection(whole_in, 16, 150, max)).show()
im.fromarray(selective_edge_detection(whole_in, 2, 5, min)).show()
#im.fromarray(selective_edge_detection(whole_in, 20, 150)).show()

output_arr = maxpool_manual(input_arr)
output_arr_rgb = [[[x,x,x] for x in line] for line in output_arr]

outputted_arr = avgpool_manual(output_arr)
assert np.any(output_arr > 1)

FREQMAP = freqmap_manual(input_arr)
#FREQMAP = freqmap_manual_log(input_arr)

iamge = im.fromarray(FREQMAP)
#iamge.show()
#iamge.save(os.path.join(DIR_IMAGESET, "test.png"))

#print(output_arr)
#print(type(output_arr))
#imwrite(os.path.join(DIR_IMAGESET, "test.tiff"), output_arr_rgb)
imwrite(os.path.join(DIR_IMAGESET, "test_chaos.tiff"), np.random.randint(0, 255, (64, 64), 'uint8'))
print("Done!")
#assert 1 == 0

def freqshow(dir, imgname):
    im.fromarray(freqmap_manual(np.array([x for x in tf.keras.utils.img_to_array(tf.keras.utils.load_img(os.path.join(DIR_IMAGESET, dir, imgname)))[:,:,0]]).astype(int))).show()

#freqshow("PARTIAL", "COT1_904x_1672y.png")

# TESTS OF FREQMAP
#im.fromarray(freqmap_manual_log(whole_in)).show()
#im.fromarray(freqmap_manual_log(whole_in)).convert('RGB').save("C:\\Users\\Muroyama lab\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\BASE\\falsecolor.jpg")

