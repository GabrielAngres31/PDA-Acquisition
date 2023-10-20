import numpy as np
import math

from tifffile import imread, imwrite

import matplotlib.pyplot as plt

import os


def quickdistance(x0, y0, x1, y1):
    return np.sqrt((x1-x0)**2+(y1-y0)**2)

quickdistance(0,0,1,1)

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
    
    contacts = list(set(contacts))

    distances = [quickdistance(x0, y0, coords[0], coords[1]) for coords in contacts]
    
    zipped_list = zip(contacts, distances)
    sorted_list = sorted(zipped_list, key=lambda s: s[1])
    contacts = [i[0] for i in sorted_list]
    
    #print(contacts)
    return contacts

def contacted_pixels(contacts):
    #print(f"Contacts: {contacts}")
    pixel_list = []
    for i in range(len(contacts)-1):
        x_i, y_i, x_f, y_f = *contacts[i], *contacts[i+1]
        pixel_list.append((math.floor(min(x_i, x_f)), math.floor(min(y_i, y_f))))
    return pixel_list

def line_portions(contacts, pixel_list):
    #print(pixel_list)
    #print(contacts)
    assert len(pixel_list) == len(contacts) - 1, f"Mismatch between pixel list length {len(pixel_list)} and contact list length {len(contacts)}, which should be greater by 1."
    portions = []
    for i in range(len(pixel_list)):
        x_i, y_i, x_f, y_f = *contacts[i], *contacts[i+1]
        line_length = quickdistance(x_i, y_i, x_f, y_f)
        portions.append(line_length)
    return portions

def contact_distances(coords_test, target):
    return [quickdistance(*coords, *target) for coords in coords_test]

pos0_TEST = (0,0)
pos1_TEST = (3,2)

coords_TEST = [*pos0_TEST, *pos1_TEST]

LATTICE_CONTACTS = lattice_contacts(*coords_TEST)
print(LATTICE_CONTACTS)
print(len(LATTICE_CONTACTS))

CONTACTED_PIXELS = contacted_pixels(LATTICE_CONTACTS)
print(CONTACTED_PIXELS)
print(len(CONTACTED_PIXELS))

LINE_PORTIONS = line_portions(LATTICE_CONTACTS, CONTACTED_PIXELS)
print(LINE_PORTIONS)


# C:\Users\gjang\Pictures
DIR_MAIN = os.getcwd()
DIR_TEST_IMAGE = os.path.join(DIR_MAIN, "TEST_IMAGES")
im_in = imread(os.path.join(DIR_TEST_IMAGE, "TESTING.tiff"))
im_in_part = imread(os.path.join(DIR_TEST_IMAGE, "TEST_PART.tiff"))
im_in_non = imread(os.path.join(DIR_TEST_IMAGE, "TEST_NON.tiff"))
im_in_scr = imread(os.path.join(DIR_TEST_IMAGE, "scrunch.tiff"))
im_in_edge = imread(os.path.join(DIR_TEST_IMAGE, "EDGE.tiff"))

im_in_stem = imread(os.path.join(DIR_TEST_IMAGE, "STEM.tiff"))
im_in_crowd = imread(os.path.join(DIR_TEST_IMAGE, "CROWD.tiff"))


im_in = np.asarray([item for item in [[pixel[0] for pixel in line] for line in im_in]])
im_in_part = np.asarray([item for item in [[pixel[0] for pixel in line] for line in im_in_part]])
im_in_non = np.asarray([item for item in [[pixel[0] for pixel in line] for line in im_in_non]])
im_in_scr = np.asarray([item for item in [[pixel[0] for pixel in line] for line in im_in_scr]])
im_in_edge = np.asarray([item for item in [[pixel[0] for pixel in line] for line in im_in_edge]])

im_in_stem = np.asarray([item for item in [[pixel[0] for pixel in line] for line in im_in_stem]])
im_in_crowd = np.asarray([item for item in [[pixel[0] for pixel in line] for line in im_in_crowd]])
#print(im_in)

def coord_to_position(x, y, width):
        return x+y*width

def position_to_coord(position, width):
        return (position % width, position//width)

def remap_radial_kernel(arr, val = 0, remove_axes = False):
    height = arr.shape[1]
    width = arr.shape[0]

    size = arr.size

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
            return_val = [max(values), min(values), int(np.mean(contributions))][val]
            #print( return_val )
            return return_val
            #print(arr_out[h,w])
            #print(arr_out)

    arr_out_flat = [linechecker(arr, pos) for pos in range(arr.size)]
    arr_out = np.reshape(arr_out_flat, arr.shape)

    if remove_axes:
        arr_out = np.delete(arr_out, 25, 1)
        arr_out = np.delete(arr_out, 25, 0)

    # for h in range(height):
    #     for w in range(width):
    #         if w == center[0] or h == center[1]:
    #             print(h)
    #             print(w)
    #             print(arr_out[h,w])
    #             arr_out[h, w] = 255
    #             print(arr_out[h,w])
                
    #             continue
            #print(arr_out[h,w])
            #print(*center, w, h)
            ###endpoints = [*center, w, h]
            #print(f"Endpoints: {endpoints[0], endpoints[1]}, {endpoints[2], endpoints[3]}")
            ###contacts = lattice_contacts(*endpoints)
            #print(f"{len(contacts)} Contacts: {contacts}")
            ###pixels = contacted_pixels(contacts)
            #print(f"{len(pixels)} Pixels: {pixels}")
            #assert len(pixels) == len(contacts) - 1, f"Mismatch between pixel list length {len(pixels)} and contact list length {len(contacts)}, which should be greater by 1."
            ###portions = line_portions(contacts, pixels)
            ###values = [arr[px[1], px[0]] for px in pixels]
            ###assert len(values) == len(portions), f"Mismatch between pixel list length {len(values)} and portion list length {len(portions)}, which should be equal."

            ###contributions = [v*p for v,p in zip(values, portions)]
            #print(int(np.floor(np.mean(contributions))))
            #arr_out[h, w] = int(np.floor(np.mean(contributions)))
            #print(max(values))
            #arr_out[h, w] = max(values)
            #print("Giges")
            ###arr_out[h,w] = 255
            #print(arr_out[h,w])
            #print(arr_out)
    #print(arr_out)
    return arr_out
#im_out = np.asarray([[[value, value, value] for value in line] for line in im_out])

def Pool_Manual_2x2(arr, func):
    assert arr.shape[0] % 2 == 0, arr.shape[1] % 2 == 0
    w = arr.shape[1]
    arr_flat = arr.flatten()
    #arr_out_flat = [ 
    #    np.mean(arr_flat [[
    #        2*n +   w  *(2*n//w), 2*n +   w  *(2*n//w) + 1,
    #        2*n + (w+1)*(2*n//w), 2*n + (w+1)*(2*n//w) + 1,
    #]]) for n in range(arr.size//4)
    #]

    arr_out_flat = [
         func(arr_flat [[
            2*n + 2*  w * (n//w),    2*n + 2* w * (n//w) + 1,
            2*n + 2*  w *((n//w)+1), 2*n + 2* w *((n//w) + 1) + 1,
    ]]) for n in range(arr.size//4)
    ]    
    
    #print(arr_out_flat)
    arr_out = np.array(arr_out_flat).astype(int).reshape(tuple([d//2 for d in arr.shape]))
    return arr_out

#im_out = np.asarray([[[*map(int, [value]*3)] for value in line] for line in im_out])
#assert 1 == 0
#imwrite("C:\\Users\\gjang\\Pictures\\RESULT.tiff", im_out, photometric = 'grayscale')

def symmetry_remap(arr):
    assert arr.shape[0] == arr.shape[1]
    parity = arr.size % 2
    center = arr.shape[0]//2
    arr_out = np.full(arr.shape, 0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            #print(i, j)
            if i == j == center and parity:
                continue
            #print(255-abs(arr[i,j] - arr[-i-1,-j-1]))
            arr_out[i, j] = 255-int(abs(arr[i,j] - arr[-i-1,-j-1]))
    if parity:
        arr_out[center, center] = int(np.mean(list((
            arr_out[center-1, center-1],
            arr_out[center  , center-1],
            arr_out[center+1, center-1],
            arr_out[center-1, center  ],
            # This would be the center pixel
            arr_out[center+1, center  ],
            arr_out[center-1, center+1],
            arr_out[center  , center+1],
            arr_out[center+1, center+1],
        ))))
        
    return arr_out

def ring(arr, depth):
    for i in arr[depth]:
        yield i
    for i in range(depth+1, len(arr)-1-depth):
        yield arr[i][depth]
        yield arr[i][-1-depth]
    for i in arr[-1-depth]:
        yield i

def center_area_homogeneity_remap(arr, func, choice = "square"):
    assert arr.shape[0] == arr.shape[1]
    arr_out = np.full(arr.shape, 128)
    side = arr.shape[0]
    if choice == "square":
        for l in range(side//2):
            if l == side - l:
                continue
            arr_out[l:side-l, l:side-l] = int(func(arr[l:side-l, l:side-l]))
    if choice == "ring":
        for d in range(side//2):
            ring_elements = [i for i in ring(arr, d)]
            value = int(func(ring_elements))
            arr_out[d] = value
            arr_out[-1-d] = value
            for p in range(d+1, side-1-d):
                arr_out[p][d] = value
                arr_out[p][-1-d] = value
    
    return arr_out

def abs_subtractive_convolve_remap(arr_base, arr_mask):
    assert arr_base.shape == arr_mask.shape
    arr_out = np.abs(np.subtract(arr_mask, arr_base))
    #arr_out = np.subtract(np.square(arr_mask), np.square(arr_base))
    #np.place(arr_out, arr_out<0, 0)
    #arr_out = np.sqrt(arr_out).astype(int)
    return arr_out

def floor_subtractive_convolve_remap(arr_base, arr_mask):
    assert arr_base.shape == arr_mask.shape
    arr_out = np.subtract(arr_mask, arr_base)
    np.place(arr_out, arr_out<0, 0)
    return arr_out

def invert(arr):
    return np.subtract(255, arr)

def pass_filter(arr, threshold, highpass = True):
    
    arr_out = np.full(arr.shape, 0)

    if highpass:
        np.copyto(arr_out, arr, where = arr>threshold)
    else:
        np.copyto(arr_out, arr, where = arr<threshold)
    return arr_out

def filter_proportion_threshold(arr, threshold, highpass = True):
    arr_flat = np.copy(arr).flatten()
    np.delete(arr_flat, arr_flat == 0)
    thresh_int = int(np.quantile(arr_flat, threshold))
    return pass_filter(arr, thresh_int, highpass)




arr_test = np.arange(81).reshape((9,9))
print(arr_test)
print(pass_filter(arr_test, 25))



def UNIT_TEST_plotter():
    removeaxes = True
    files = (im_in, im_in_non, im_in_part, im_in_scr, im_in_edge, im_in_stem, im_in_crowd)
    plt.figure(1)
    f = len(files)
    for i,file in enumerate(files):
        
        base_radial_kernel = remap_radial_kernel(file, 0, removeaxes)
        symm_radial_kernel = symmetry_remap(base_radial_kernel)
        diff_radial_kernel = floor_subtractive_convolve_remap(base_radial_kernel, symm_radial_kernel)
        crck_radial_kernel = floor_subtractive_convolve_remap(base_radial_kernel, diff_radial_kernel)
        threshold_r_kernel = filter_proportion_threshold(crck_radial_kernel, 0.15)
        abs__thresh_kernel = pass_filter(crck_radial_kernel, 108)
        prop_thresh_kernel = filter_proportion_threshold(abs__thresh_kernel, 0.6)
        maxd_cnvlve_kernel = Pool_Manual_2x2(prop_thresh_kernel[1:, 1:], np.max)

        plot_objects = (file, base_radial_kernel, symm_radial_kernel, diff_radial_kernel, crck_radial_kernel, abs__thresh_kernel, prop_thresh_kernel, maxd_cnvlve_kernel)
        p = len(plot_objects)
        for j,plot_obj in enumerate(plot_objects):
            plt.subplot(p, f, i+f*j+1)
            plt.imshow(plot_obj, cmap='gray')

    plt.show()

UNIT_TEST_plotter()


#for i in range(6):
#    w = i // 3
#    print(f"{2*i+6*w}, {2*i+6*w+1}, {6*(w)+6+2*i}, {6*w+6+2*i+1}")

#  0   1   2   3   4   5
#  6   7   8   9  10  11
# 12  13  14  15  16  17
# 18  19  20  21  22  23




#print(MeanPool_Manual_2x2(arr_test))
#print(MeanPool_Manual_2x2(im_in))
