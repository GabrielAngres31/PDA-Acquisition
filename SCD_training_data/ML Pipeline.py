import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import sys
from tifffile import imread, imwrite
from PIL import Image as im
import numpy as np



DIR_CWD = os.getcwd()
DIR_SOURCE = os.path.join(DIR_CWD, "source_images")
DIR_IMAGESET = os.path.join(DIR_SOURCE, "generated\\TOTALS")
DIR_ABSENT = os.path.join(DIR_IMAGESET, "ABSENT") 
DIR_PARTIAL= os.path.join(DIR_IMAGESET, "PARTIAL")
DIR_WHOLE  = os.path.join(DIR_IMAGESET, "WHOLE")

image_dataset = tf.keras.utils.image_dataset_from_directory(DIR_IMAGESET, 
                                           color_mode="grayscale",
                                           validation_split=0.4,
                                           subset = "both",
                                           seed = 3735928559,
                                           label_mode = "categorical",
                                           image_size = (1, 64, 64),
                                           labels = 'inferred')



print(image_dataset)

def coord_to_position(x, y, width):
        return x+y*width

def position_to_coord(position, width):
        return (position % width, position//width)

def quickdistance(x0, y0, x1, y1):
    return np.sqrt((x1-x0)**2+(y1-y0)**2)

def lattice_contacts(x0, y0, x1, y1):
    #print(locals())
    assert(all(map(lambda n: n % 1 == 0, [x0, y0, x1, y1])))

    contacts = []
    delta_x, delta_y = int(x1-x0), int(y1-y0)
    delta_D = max(abs(delta_x), abs(delta_y))
    
    assert delta_x != 0, "Vertical Lines are Not Allowed!"
    assert delta_y != 0, "Horizontal Lines are Not Allowed!"
    m = delta_y/delta_x
    xs, ys = np.sign(delta_x), np.sign(delta_y)

    for x in range(x0, x1+xs, xs):
        #print(f"x: {x}")
        #print(((x0+x)*1.0, m*x + y0))
        contacts.append((x, m*(x-x0)+y0))
    for y in range(y0, y1+ys, ys):
        #print(f"y: {y}")
        #print(((y+y0)/m, (y+y0)*1.0))
        contacts.append(((y-y0)/m+x0, y))

    ###
    # print(contacts)
    # contacts.append([(x, m*(x-x0+y0)) for x in range(x0, x1+xs, xs)])
    # print(contacts)
    # contacts.append([((y-y0)/m+x0, y) for y in range(y0, y1+xs, ys)])
    # print(contacts)
    ###
    # flatten_list = lambda y:[x for a in y for x in flatten_list(a)] if type(y) is list else [y]
    # contacts = flatten_list(contacts)
    contacts = list(set(contacts))

    distances = [quickdistance(x0, y0, coords[0], coords[1]) for coords in contacts]
    
    zipped_list = zip(contacts, distances)
    sorted_list = sorted(zipped_list, key=lambda s: s[1])
    contacts = [i[0] for i in sorted_list]
    
    #print(contacts)
    return contacts

def contacted_pixels(contacts):
    #print(f"Contacts: {contacts}")
    #pixel_list = []
    # for i in range(len(contacts)-1):
    #     x_i, y_i, x_f, y_f = *contacts[i], *contacts[i+1]
    #     pixel_list.append((math.floor(min(x_i, x_f)), math.floor(min(y_i, y_f))))
    ###
    return [(math.floor(min(contacts[i][0], contacts[i+1][0])), math.floor(min(contacts[i][1], contacts[i+1][1]))) for i in range(len(contacts)-1)]
    ###

def line_portions(contacts, pixel_list):
    #print(pixel_list)
    #print(contacts)
    assert len(pixel_list) == len(contacts) - 1, f"Mismatch between pixel list length {len(pixel_list)} and contact list length {len(contacts)}, which should be greater by 1."
    #portions = []
    #for i in range(len(pixel_list)):
    #    x_i, y_i, x_f, y_f = *contacts[i], *contacts[i+1]
    #    line_length = quickdistance(x_i, y_i, x_f, y_f)
    #    portions.append(line_length)
    return [quickdistance(*contacts[i], *contacts[i+1]) for i in range(len(pixel_list))]

def contact_distances(coords_test, target):
    return [quickdistance(*coords, *target) for coords in coords_test]

def remap_radial_kernel(arr, val = 0, remove_axes = False):
    height = arr.shape[1]
    width = arr.shape[0]

    #size = arr.size

    assert width % 2 == 0, height % 2 == 0

    arr_out = np.full(arr.shape, 0, dtype = int)
    #print(arr_out)

    center = [width//2, height//2]
    
    def linechecker(arr_in, pos):
        interpolation = int(np.mean(arr_in))
        coord = position_to_coord(pos, width)
        if coord[0] == center[0] or coord[1] == center[1]:
            return interpolation
        else:
            endpoints = [*center, coord[0], coord[1]]
            
            contacts = lattice_contacts(*endpoints)
            
            pixels = contacted_pixels(contacts)
            #assert not(any(x>arr_in.shape[0] for x in pixels))
            #assert len(pixels) == len(contacts) - 1, f"Mismatch between pixel list length {len(pixels)} and contact list length {len(contacts)}, which should be greater by 1."
            portions = line_portions(contacts, pixels)
            values = [arr_in[px[1], px[0]] for px in pixels]
            assert len(values) == len(portions), f"Mismatch between pixel list length {len(values)} and portion list length {len(portions)}, which should be equal."

            contributions = [v*p for v,p in zip(values, portions)]
            #print(int(np.floor(np.mean(contributions))))
            #arr_out[h, w] = int(np.floor(np.mean(contributions)))
            #print(max(values))
            #arr_out[h, w] = max(values)
            #print("Giges")
            return [max(values), min(values), int(np.mean(contributions))][val]
            #print( return_val )
            
            #print(arr_out[h,w])
            #print(arr_out)

    arr_out_flat = [linechecker(arr, pos) for pos in range(arr.size)]
    arr_out = np.reshape(arr_out_flat, arr.shape)

    if remove_axes:
        arr_out = np.delete(arr_out, center, 1)
        arr_out = np.delete(arr_out, center, 0)

    return arr_out

inputs = keras.Input(shape=(64,64))
# x = remap_radial_kernel(inputs)
x = layers.Rescaling(1./255)(inputs)