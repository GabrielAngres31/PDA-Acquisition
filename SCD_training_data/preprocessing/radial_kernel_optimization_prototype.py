import numpy as np

import math 

from tifffile import imread, imwrite
import matplotlib.pyplot as plt
import os


def coord_to_position(x, y, width):
        return x+y*width

def position_to_coord(position, width):
        return (position // width, position % width)

def quickdistance(x0, y0, x1, y1):
    return np.sqrt((x1-x0)**2+(y1-y0)**2)


def lattice_contacts(x0, y0, x1, y1):
    
    assert(all(map(lambda n: n % 1 == 0, [x0, y0, x1, y1])))

    contacts = []
    delta_x, delta_y = int(x1-x0), int(y1-y0)
    delta_D = max(abs(delta_x), abs(delta_y))
    
    assert delta_x != 0, "Vertical Lines are Not Allowed!"
    assert delta_y != 0, "Horizontal Lines are Not Allowed!"
    m = delta_y/delta_x
    xs, ys = np.sign(delta_x), np.sign(delta_y)
    
    contacts_x = [(x, m*(x-x0)+y0) for x in range(x0, x1+xs, xs)]
        
    # for y in range(y0, y1+ys, ys):
    #     contacts.append(((y-y0)/m+x0, y))

    contacts_y = [((y-y0)/m+x0, y) for y in range(y0, y1+ys, ys)]

    contacts.extend(contacts_x)
    contacts.extend(contacts_y)

    contacts = list(set(contacts))

    distances = [quickdistance(x0, y0, coords[0], coords[1]) for coords in contacts]
    
    zipped_list = zip(contacts, distances)
    sorted_list = sorted(zipped_list, key=lambda s: s[1])
    contacts = [i[0] for i in sorted_list]
    
    return contacts

def contacted_pixels(contacts):
    return [(math.floor(min(contacts[i][0], contacts[i+1][0])), math.floor(min(contacts[i][1], contacts[i+1][1]))) for i in range(len(contacts)-1)]


def line_portions(contacts, pixel_list):
    assert len(pixel_list) == len(contacts) - 1, f"Mismatch between pixel list length {len(pixel_list)} and contact list length {len(contacts)}, which should be greater by 1."
    return [quickdistance(*contacts[i], *contacts[i+1]) for i in range(len(pixel_list))]

def contact_distances(coords_test, target):
    return [quickdistance(*coords, *target) for coords in coords_test]

def linecheck_ref_array_max(length):

    arr_sample_num = np.arange(length**2).reshape((length, length))
    arr_output_lst = list()
    center = [length//2, length//2]
    recode_center = center[::-1]

    def linechecker_max(arr_in, pos):
        
        coord = list(position_to_coord(pos, length))
        
        recode_coord = coord[::-1]

        #index_vert = lambda a, r0, r1, c: a[   r0:r1:np.sign(r1-r0), c]
        #index_horz = lambda a, c0, c1, r: a[r, c0:c1:np.sign(c1-c0)   ]

        y, x  = recode_coord[0], recode_coord[1]
        yc, xc = recode_center[0], recode_center[1]
        if not(recode_coord[0] == recode_center[0]):
            if not(recode_coord[1] == recode_center[1]):
                endpoints = [*center, coord[0], coord[1]]
                contacts = lattice_contacts(*endpoints)
                pixels = contacted_pixels(contacts)
                return [arr_in[px[1], px[0]] for px in pixels]
            else:
                return [arr_in[k][x] for k in range(min(y, yc), max(y, yc)+1)]
        else:
            return [arr_in[y][k] for k in range(min(x, xc), max(x, xc)+1)]
    arr_output_lst = [linechecker_max(arr_sample_num, i) for i in range(length**2)]
    return arr_output_lst


def norm_brighten(arr):

    range_old = np.max(arr) - np.min(arr)

    arr_out_float = (arr - np.min(arr)) * (255 / range_old)
    arr_out = arr_out_float.astype("int")
    return arr_out

def invert(arr):
    return np.subtract(255, arr)

def pass_filter(arr, threshold, highpass = True):
    
    arr = arr.astype("int")
    arr_out = np.full(arr.shape, 0)

    if highpass:
        np.copyto(arr_out, arr, where = arr>threshold)
    else:
        np.copyto(arr_out, arr, where = arr<threshold)
    return arr_out

def floor_subtractive_convolve_remap(arr_base, arr_mask):
    assert arr_base.shape == arr_mask.shape
    arr_out = np.subtract(arr_mask, arr_base)
    #assert np.all(len(arr_out) == 1), f"{arr_out}"
    np.place(arr_out, arr_out<0, 0)
    return arr_out

def Pool_Manual_2x2(arr, func):

    arr_np = np.array(arr)
    assert arr_np.shape[0] % 2 == 0, arr_np.shape[1] % 2 == 0
    w = arr_np.shape[1]
    arr_flat = arr_np.flatten()
    arr_out_flat = [
         func(arr_flat [[
            2*n + 2*  w * (n//w),    2*n + 2* w * (n//w) + 1,
            2*n + 2*  w *((n//w)+1), 2*n + 2* w *((n//w) + 1) + 1,
    ]]) for n in range(arr.size//4)
    ]    
    arr_out = np.array(arr_out_flat).reshape(arr.shape[0]//2, arr.shape[1]//2)
    return arr_out

        
def flip_xaxe(point: tuple, size: int):
    y,x = point[0], point[1] 
    return tuple(y, size-1-x)

def flip_yaxe(point: tuple, size: int):
    y,x = point[0], point[1] 
    return tuple(size-1-y, x)

def flip_orig(point: tuple, size: int):
    y,x = point[0], point[1]
    return tuple(size-1-y, size-1-x)

def flip_vert_flat(pos: int, width: int):
    coord = position_to_coord(pos, width) 
    y, x = coord[0], coord[1]
    return (width-1)-x + width*y

def flip_horz_flat(pos: int, width: int):
    coord = position_to_coord(pos, width) 
    y, x = coord[0], coord[1]
    return x + width*(width-1-y)

def flip_orig_flat(pos: int, width: int):
    coord = position_to_coord(pos, width) 
    y, x = coord[0], coord[1]
    return (width-1)-x + width*(width-1-y)

def flip_xeqy_flat(pos: int, width: int): # Produces a flip across x=y
    coord = position_to_coord(pos, width) 
    y, x = coord[0], coord[1]
    return (width+1)*(x+y)-pos

def kernel_lookup_list(width):
    

    flatten_list = lambda y:[x for a in y for x in flatten_list(a)] if type(y) is list else [y]
    
    if width % 2:
        pass
    else:
        quad_init_pos = [[coord_to_position(*coord, width) for coord in contacted_pixels(lattice_contacts(y,x,width//2,width//2))] for x in range(width//2) for y in range(width//2)]
        
        quad_init_num = [i%(width//2)+i//(width//2)*width for i in range((width//2)**2)]
        #return quad_init
        #print("v")
        quad_pos_v_flip = [[flip_vert_flat(pos, width) for pos in ind] for ind in quad_init_pos]
        quad_num_v_flip = [flip_vert_flat(num, width) for num in quad_init_num]

        #print("h")
        quad_pos_h_flip = [[flip_horz_flat(pos, width) for pos in ind] for ind in quad_init_pos]
        quad_num_h_flip = [flip_horz_flat(num, width) for num in quad_init_num]
        #print("o")
        quad_pos_o_flip = [[flip_orig_flat(pos, width) for pos in ind] for ind in quad_init_pos]
        quad_num_o_flip = [flip_orig_flat(num, width) for num in quad_init_num]

        total_pos = quad_init_pos
        total_num = quad_init_num
        total_pos.extend(quad_pos_v_flip)
        total_pos.extend(quad_pos_h_flip)
        total_pos.extend(quad_pos_o_flip)
        total_num.extend(quad_num_v_flip)
        total_num.extend(quad_num_h_flip)
        total_num.extend(quad_num_o_flip)

        zipped = zip(total_num, total_pos)
        flat_zip_sort = sorted(zipped, key = lambda x: x[0])
        list_out = [piece[1] for piece in flat_zip_sort]
        return list_out


def maxcalc_from_kernel(arr_in, kernel):
    
    def setarray_max(arr, kern, pos):
        return max([arr[position_to_coord(p, arr.shape[0])] for p in kern[pos]])

    arr_out = np.array([setarray_max(arr_in, kernel, pos) for pos in range(arr_in.size)]).reshape(arr_in.shape)
    
    assert len(kernel) == arr_in.size, f"Sizes mismatch! Flat Kernel: {len(kernel)}, Array Size: {arr_in.size}"
    # arr_in_flat = arr_in.flatten()
    # flat_list_out = [max([arr_in_flat[pos] for pos in lookup_pos]) for lookup_pos in kernel]
    # arr_out = np.array(flat_list_out).reshape(arr_in.shape)

    return arr_out


def UNIT_TEST_plotter():
    DIR_MAIN = os.getcwd()
    DIR_TEST_IMAGE = os.path.join(DIR_MAIN, "TEST_IMAGES")

    file_names = ("TESTING", "TEST_PART", "TEST_NON", "scrunch", "EDGE", "STEM", "CROWD")
    file_loads = (imread(os.path.join(DIR_TEST_IMAGE, file_name + ".tiff")) for file_name in file_names)
    files = (np.asarray([item for item in [[pixel[0] for pixel in line] for line in file_load]]) for file_load in file_loads)
    
    plt.figure(1)
    f = len(file_names)
    for i,file in enumerate(files):
        
        base_radial_kernel = maxcalc_from_kernel(file, kernel_lookup_list(50))

        plot_objects = (file, base_radial_kernel)
        p = len(plot_objects)
        for j,plot_obj in enumerate(plot_objects):
            plt.subplot(p, f, i+f*j+1)
            plt.imshow(plot_obj, cmap='gray', vmin = 0, vmax = 255)

    plt.show()

#UNIT_TEST_plotter()
        
CANVAS = np.full((8, 8), 255)

def drawoncanvas(canvas, endpos):
    canvas_flat = canvas.flatten()
    lookup = kernel_lookup_list(canvas.shape[0])
    canvas_flat[lookup[endpos]] = 0
    #plt.imshow(canvas_flat.reshape(canvas.shape), cmap = 'gray', vmin = 0, vmax = 255)
    #plt.show()
    return canvas_flat.reshape(canvas.shape)

# plt.figure(1)
# for i in range(8**2):
#     plt.subplot(8, 8, i+1)
#     plt.imshow(drawoncanvas(CANVAS, i), cmap = 'gray', vmin = 0, vmax = 255)
# plt.show()



def RANDOM_SAMPLE_plotter(size, num, pool_path = "cot1.tif", mask_pool_path = "cot1_STOM_BIN_MASK.tiff"):
    source_path = "C:\\Users\\Muroyama lab\\Documents\\Muroyama Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images"
    #source_path = "C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images"
    file_path = os.path.join(source_path, "BASE", pool_path)
    mask_path = os.path.join(source_path, "ANNOTATION", mask_pool_path)
    file_array = imread(file_path)
    mask_array = imread(mask_path)

    def select_section(x, y, size):
        return file_array[y:y+size, x:x+size]


    def section_generator(num, arr, stomata = False, size = size):
        i = 0
        w = arr.shape[1]
        h = arr.shape[0]
        while i < num:
            interesting = False
            while not(interesting):
                if stomata:
                    x, y = np.random.randint(w-size), np.random.randint(h-size)
                    if not(np.any(mask_array[y+size//2, x+size//2] == 0)):
                        interesting = True
                else:
                    x, y = np.random.randint(w-size), np.random.randint(h-size)
                    if np.mean(select_section(x, y, size)) > 4:
                        interesting = True
            yield select_section(x, y, size)
            i += 1
    plt.figure(1)
    files = [i for i in section_generator(num, file_array, stomata = True)]
    f = len(files)
    SAMPLE_KERNEL = kernel_lookup_list(size)
    for i,file in enumerate(files):
        base = maxcalc_from_kernel(file, SAMPLE_KERNEL)
        brht = norm_brighten(base)
        invt = invert(brht)
        flrd = floor_subtractive_convolve_remap(brht, invt)
        cutt = pass_filter(flrd, 200)
        overlay = np.add(file, cutt)
        
        bin_mask = np.copy(cutt)
        np.place(bin_mask, bin_mask >= 200, 255)
        #print(bin_mask)

        #maxpool_1 = Pool_Manual_2x2(bin_mask, max)
        #maxpool_2 = Pool_Manual_2x2(maxpool_1, max)
        
        plot_objects = (file, base, brht, invt, flrd, cutt, overlay, bin_mask)#, maxpool_1, maxpool_2)
    
        p = len(plot_objects)
        for j,plot_obj in enumerate(plot_objects):
            plt.subplot(p, f, i+f*j+1)
            plt.imshow(plot_obj, cmap='gray', vmin = 0, vmax = 255)
    plt.show()

RANDOM_SAMPLE_plotter(64, 12)