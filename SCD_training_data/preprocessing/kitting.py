import numpy as np
import math

from tifffile import imread, imwrite

import matplotlib.pyplot as plt

import os
import tqdm

from multiprocessing import Pool

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

    #contacts = [(x, m*(x-x0)+y0) for x in range(x0, x1+xs, xs)] + [((y-y0)/m+x0, y) for y in range(y0, y1+ys, ys)]

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

pos0_TEST = (0,0)
pos1_TEST = (3,2)

coords_TEST = [*pos0_TEST, *pos1_TEST]

#LATTICE_CONTACTS = lattice_contacts(*coords_TEST)
#print(LATTICE_CONTACTS)
#print(len(LATTICE_CONTACTS))

#CONTACTED_PIXELS = contacted_pixels(LATTICE_CONTACTS)
#print(CONTACTED_PIXELS)
#print(len(CONTACTED_PIXELS))

#LINE_PORTIONS = line_portions(LATTICE_CONTACTS, CONTACTED_PIXELS)
#print(LINE_PORTIONS)


# C:\Users\gjang\Pictures

#print(im_in)

def coord_to_position(x, y, width):
        return x+y*width

def position_to_coord(position, width):
        return (position % width, position//width)

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
        if coord[0] == center[0] and coord[1] == center[1]:
            return arr_in[center[1], center[0]]
        elif coord[0] == center[0]:
            #return max((lambda r, a_b, c: r[min(a_b):max(a_b), c])(arr_in, [center[1], coord[1]], center[0]))
            #return arr_in[coord[1], coord[0]]
            return 64
        elif coord[1] == center[1]:
            #return max((lambda r, a_b, c: r[c, min(a_b):max(a_b)])(arr_in, [center[0], coord[0]], center[1]))
            #return arr_in[coord[1], coord[0]]
            return 192
        #elif coord[0] == center[0] or coord[1] == center[1]:
        #    return 0
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

def symmetry_remap(arr, func = lambda x: np.subtract(255, np.abs(x - np.flip(x)))):
    assert arr.shape[0] == arr.shape[1]
    parity = arr.size % 2
    center = arr.shape[0]//2
    arr_out = np.full(arr.shape, 0)
    #for i in range(arr.shape[0]):
    #    for j in range(arr.shape[1]):
            #print(i, j)
    #        if i == j == center and parity:
    #            continue
            #print(255-abs(arr[i,j] - arr[-i-1,-j-1]))
            #arr_out[i, j] = 255-int(abs(arr[i,j] - arr[-i-1,-j-1]))

    
    arr_out = func(arr)
    
    if parity:
        arr_out[center, center] = int(
            np.mean(
                np.delete(
                    arr_out[center-1:center+2, center-1:center+2,].flatten(), 4
                )
            )
        )
        
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

def norm_brighten(arr, factor = 4):
    # minimum = np.min(arr)
    # maximum = np.max(arr)
    # if minimum == maximum:
    #     return np.full(arr.shape, 0)
    # else:
    #     delta = maximum-minimum
    #     return factor*255*np.subtract(arr, minimum)/np.subtract(maximum, minimum)
    
    multed = np.multiply(np.subtract(arr, 16), factor)
    np.place(multed, multed > 255, 255)
    return multed


arr_test = np.arange(81).reshape((9,9))
print(arr_test)
print(pass_filter(arr_test, 25))



def UNIT_TEST_plotter():
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
    removeaxes = False
    files = (im_in, im_in_non, im_in_part, im_in_scr, im_in_edge, im_in_stem, im_in_crowd)
    plt.figure(1)
    f = len(files)
    for i,file in enumerate(files):
        
        base_radial_kernel = remap_radial_kernel(file, 0, removeaxes) if removeaxes else remap_radial_kernel(file, 0, removeaxes)[1:, 1:]
        symm_radial_kernel = symmetry_remap(base_radial_kernel)
        diff_radial_kernel = floor_subtractive_convolve_remap(4*base_radial_kernel, symm_radial_kernel)
        crck_radial_kernel = floor_subtractive_convolve_remap(base_radial_kernel, diff_radial_kernel)
        threshold_r_kernel = filter_proportion_threshold(diff_radial_kernel, 0.15)
        abs__thresh_kernel = pass_filter(diff_radial_kernel, 64)
        prop_thresh_kernel = filter_proportion_threshold(abs__thresh_kernel, 0.6)
        maxd_cnvlve_kernel = Pool_Manual_2x2(diff_radial_kernel if removeaxes else diff_radial_kernel [1:, 1:], np.max)
        doub_cnvlve_kernel = Pool_Manual_2x2(maxd_cnvlve_kernel, np.max)

        plot_objects = (file, base_radial_kernel, symm_radial_kernel, diff_radial_kernel, maxd_cnvlve_kernel, doub_cnvlve_kernel)
        p = len(plot_objects)
        for j,plot_obj in enumerate(plot_objects):
            plt.subplot(p, f, i+f*j+1)
            plt.imshow(plot_obj, cmap='gray', vmin = 0, vmax = 255)

    plt.show()

UNIT_TEST_plotter()



def RANDOM_SAMPLE_plotter(size, num, pool_path = "cot1.tif", mask_pool_path = "cot1_STOMATA_MASKS.tiff"):
    #source_path = "C:\\Users\\Muroyama lab\\Documents\\Muroyama Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images"
    source_path = "C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images"
    file_path = os.path.join(source_path, "BASE", pool_path)
    mask_path = os.path.join(source_path, "ANNOTATION", mask_pool_path)
    file_array = imread(file_path)
    mask_array = imread(mask_path)
    #removeaxes = True

    def select_section(x, y, size):
        return file_array[y:y+size, x:x+size]
    
    """def processed__img(img):
        base = remap_radial_kernel(img, 0, removeaxes)
        symm = symmetry_remap(base)
        diff = floor_subtractive_convolve_remap(4*base, symm)
        cutt = pass_filter(diff, 64)
        return cutt"""
    
    def section_generator(num, arr, size = size):
        i = 0
        w = arr.shape[1]
        h = arr.shape[0]
        while i <= num:
            interesting = False
            tries = 0
            while not(interesting):
                x, y = np.random.randint(w-size), np.random.randint(h-size)
                if np.mean(select_section(x, y, size)) > 4:
                   interesting = True
                # x, y = np.random.randint(w-size), np.random.randint(h-size)
                # print(mask_array[y+size//2, x+size//2])
                # if not(np.any(mask_array[y+size//2, x+size//2][0] == 0)):
                #    interesting = True
                #print(interesting)
                tries += 1
                if tries % 10000 == 0:
                    print(tries)
            yield select_section(x, y, size)
            i += 1
    plt.figure(1)
    files = [i for i in section_generator(num, file_array)]
    f = len(files)
    for i,file in enumerate(files):
        #print(file)
        #print(i)
        base = remap_radial_kernel(file, 0, True)
        brht = norm_brighten(base, factor = 2)
        invt = invert(brht)


        symm = symmetry_remap(invt)
        flrd = floor_subtractive_convolve_remap(brht, invt)

        center = size//2

        bse_ = np.delete(np.delete(file, 32, 0), 32, 1)


        overlay = np.add(bse_, flrd)
        #cutt = pass_filter(diff, 64)
        plot_objects = (file, base, brht, invt, symm, flrd, overlay)
    
        p = len(plot_objects)
        for j,plot_obj in enumerate(plot_objects):
            plt.subplot(p, f, i+f*j+1)
            plt.imshow(plot_obj, cmap='gray', vmin = 0, vmax = 255)
    plt.show()

#RANDOM_SAMPLE_plotter(64, 8)


      

#TEST_IMAGE_SPIRAL[]
#print(TEST_IMAGE_SPIRAL)
#plt.imshow(TEST_IMAGE_SPIRAL, cmap = 'gray')
#plt.show()
#plt.imshow(TEST_IMAGE_SPIRAL, cmap = 'gray', vmin = 0, vmax = 255)
#plt.imshow(symmetry_remap(TEST_IMAGE_SPIRAL), cmap = 'gray', vmin = 0, vmax=255)
#plt.show()

def IMAGE_PROCESSOR(path, num_range, kernel_size = 64):
    im_in = imread(path)
    #if len(im_in.shape) == 3:
        #image_file = np.asarray([item for item in [[pixel[0] for pixel in line] for line in image_file]])
    location = num_range[0]
    pbar = tqdm.tqdm(total = num_range[1]-num_range[0])
    while location < num_range[1]:

        h = location // (im_in.shape[0] - kernel_size)
        w = location  % (im_in.shape[1] - kernel_size)

        section = im_in[h:h+kernel_size, w:w+kernel_size]
        base_radial_kernel = remap_radial_kernel(section, 0, True)
        symm_radial_kernel = symmetry_remap(base_radial_kernel)
        diff_radial_kernel = floor_subtractive_convolve_remap(1.3*base_radial_kernel, symm_radial_kernel)

        #print(diff_radial_kernel)
        yield(diff_radial_kernel.shape)
        pbar.update(1)
        location += 1
    pbar.close()
            
#print(list(IMAGE_PROCESSOR("C:\\Users\\Muroyama lab\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\BASE\\cot1.tif", [0,1000])))


def RADIAL_CONTRAST_TALLY(image_path, size):
    
    image_file = imread(image_path)
    #print(image_file)
    if len(image_file.shape) == 3:
        image_file = np.asarray([item for item in [[pixel[0] for pixel in line] for line in image_file]])
    image_height = image_file.shape[0]
    image_width = image_file.shape[1]
    counts = np.full(256, 0)#.astype(dtype=np.int64)

    """    for h in tqdm.tqdm(range(image_height-size-1)):
        for w in range(image_width-size-1):
            symm = symmetry_remap(image_file[h:h+size, w:w+size], lambda x,y: 0 if x == y == 0 else abs(x-y))
            unique, arrcts = np.unique(symm.flatten(), return_counts=True)
            for number, count in zip(unique, arrcts):
                num_input = min(255, count) # Getting number too big error
                counts[number] += num_input
    """
    def section():
        pass
    raw_counts = (
            (
            symmetry_remap(
                image_file[h:h+size, w:w+size], lambda x: np.abs(x-np.flip(x))
                ).flatten() 
                for w in range(image_width-size-1)
            ) 
                for h in tqdm.tqdm(range(image_height-size-1))
        )
    ite = 0
    for list_gen in tqdm.tqdm(raw_counts):
        for list_pc in list_gen:
            print(list_pc)
            for item in list_pc:
                counts[item] += 1
    return(counts/np.sum(counts))
    #flatten_list = lambda y:[x for a in y for x in flatten_list(a)] if type(y) is list else [y]
    #flt_counts = flatten_list(raw_counts)
    #zip_counts = zip(np.unique(flt_counts, return_counts = True))
    #fin_counts = np.full(256, 0)
    #for val, ct in zip_counts:
    #    print(val)
    #    print(ct)
    #    fin_counts[val] = ct
    #return fin_counts

#RCT_SAMPLE = RADIAL_CONTRAST_TALLY("C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\BASE\\cot1_STUPID.tiff", 4)
#print(RCT_SAMPLE)

#labels = list(range(256))

#plt.bar(labels, RCT_SAMPLE)
#plt.show()



#for i in range(6):
#    w = i // 3
#    print(f"{2*i+6*w}, {2*i+6*w+1}, {6*(w)+6+2*i}, {6*w+6+2*i+1}")

#  0   1   2   3   4   5
#  6   7   8   9  10  11
# 12  13  14  15  16  17
# 18  19  20  21  22  23




#print(MeanPool_Manual_2x2(arr_test))
#print(MeanPool_Manual_2x2(im_in))
