import numpy as np

import math 
def coord_to_position(x, y, width):
        return x+y*width

def position_to_coord(position, width):
        return (position % width, position//width)

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

    for x in range(x0, x1+xs, xs):
        contacts.append((x, m*(x-x0)+y0))
        
    for y in range(y0, y1+ys, ys):
        contacts.append(((y-y0)/m+x0, y))

    contacts = list(set(contacts))

    distances = [quickdistance(x0, y0, coords[0], coords[1]) for coords in contacts]
    
    zipped_list = zip(contacts, distances)
    sorted_list = sorted(zipped_list, key=lambda s: s[1])
    contacts = [i[0] for i in sorted_list]
    
    #print(contacts)
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

        index_vert = lambda a, r0, r1, c: a[   r0:r1:np.sign(r1-r0), c]
        index_horz = lambda a, c0, c1, r: a[r, c0:c1:np.sign(c1-c0)   ]

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

    for i in range(length**2):
        print(i)
        arr_output_lst.append(linechecker_max(arr_sample_num, i))
    
    return arr_output_lst

        

print(linecheck_ref_array_max(5))


        