# zeta_measure


# Brooks and Grigsby, 2013
# BMC Medical Imaging
# doi:10.1186/1471-2342-13-7

import typing as tp
from PIL import Image
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt

image_path = "zeta_halfnhalf.png"
image_list = [
    "test_2clump_01.png",
    "test_2clump_02.png",
    "test_2clump_03.png",
    "test_2clump_04.png",
    "test_2clump_05.png",
    "test_3clump_01.png",
    "test_3clump_02.png",
    "test_single_01.png",
    "test_single_02.png",
    "test_single_03.png",
    "test_single_04.png",
]

img = Image.open(image_path).convert('L')  # Convert to grayscale
img_array = np.array(img)

def bresenham_line(x1, y1, x2, y2):
    """Bresenham's line algorithm to calculate pixels on a line."""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    error = dx - dy

    points = []
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * error
        if e2 > -dy:
            error -= dy
            x1 += sx
        if e2 < dx:
            error += dx
            y1 += sy
    return points


collected_means = []

# for m_y in range(img_array.shape[0]):
#     for m_x in range(img_array.shape[1]):
#         m_I = img_array[m_y, m_x]
#         for n_y in range(m_y, img_array.shape[0]): # m_y-1, 
#             for n_x in range(m_x, img_array.shape[1]): # m_x-1, 
#                 if n_y == m_y and n_x == m_x:
#                     continue
#                 if np.random.rand() < 0.95:
#                     continue
                
#                 n_I = img_array[n_y, n_x]
#                 #print(m_I)
#                 #print(n_I)
#                 d_I = n_I-m_I
#                 def dist(p_a, p_b): return np.sqrt(np.sum(np.square(np.array(p_a) -np.array(p_b)))) 
#                 L = dist((n_y, n_x), (m_y, m_x))
#                 points = bresenham_line(m_x, m_y, n_x, n_y)
#                 #print(points)
                
#                 intensities = [img_array[m_y, m_x], img_array[n_y, n_x]] + [img_array[p[1], p[0]] for p in points]
                
#                 pred_intensities = [img_array[m_y, m_x], img_array[n_y, n_x]] + [m_I + (n_I-m_I)*dist((m_y, m_x), p)/L for p in points]
#                 assert len(intensities) == len(pred_intensities)

#                 mean_I_deltas = sum([np.abs(pred_intensities[i] - intensities[i]) for i in range(len(intensities))])/len(intensities)

#                 collected_means.append((L, mean_I_deltas))

#                 # print(mean_I_deltas, L)
                
#                 #print(f"{img_array[m_y,m_x]}, {img_array[n_y, n_x]}")

# width, height = img_array.size
print(img_array.size)

import datetime
import tqdm
import sys
# then = datetime.datetime.now()

def calculate_zeta(img, proportion = 0.05, name="toast"):
    for m in tqdm.tqdm(range(img.size)):
        if np.random.rand() < 1-proportion:
            continue
        #print(img.shape)
        m_y, m_x = divmod(m, img.shape[1])

        m_top_bot = m_y//(img.shape[0]//2)
        m_left_right = m_x//(img.shape[1]//2)

        m_vert_step = 1 if m_top_bot else -1
        m_horz_step = 1 if m_left_right else -1

        m_vert_pix = [(s, m_x) for s in list(range(m_y, img.shape[0]*m_top_bot, m_vert_step))]
        m_vert_cal = [True if img[p_y, p_x] else False for p_y, p_x in m_vert_pix]
        
        m_horz_pix = [(m_y, s) for s in list(range(m_x, img.shape[0]*m_left_right, m_horz_step))]
        m_horz_cal = [True if img[p_y, p_x] else False for p_y, p_x in m_horz_pix]

        if any(m_vert_cal) and any(m_horz_cal):
            continue
        for n in range(m+1, img.size):
            
            n_y, n_x = divmod(n, img.shape[1])

            n_top_bot = n_y//(img.shape[0]//2)
            n_left_right = n_x//(img.shape[1]//2)

            n_vert_step = 1 if n_top_bot else -1
            n_horz_step = 1 if n_left_right else -1

            n_vert_pix = [(s, n_x) for s in list(range(n_y, img.shape[0]*n_top_bot, n_vert_step))]
            n_vert_cal = [True if img[p_y, p_x] else False for p_y, p_x in n_vert_pix]
            
            n_horz_pix = [(m_y, s) for s in list(range(n_x, img.shape[0]*n_left_right, n_horz_step))]
            n_horz_cal = [True if img[p_y, p_x] else False for p_y, p_x in n_horz_pix]

            if any(n_vert_cal) and any(n_horz_cal):
                continue

            m_I = img[m_y, m_x]
            #print(np.random.rand())
            
            n_I = img[n_y, n_x]
            #print(m_I)
            #print(n_I)
            d_I = n_I-m_I
            def dist(p_a, p_b): return np.sqrt(np.sum(np.square(np.array(p_a) - np.array(p_b)))) 
            L = dist((n_y, n_x), (m_y, m_x))
            if L < 20:
                continue
            points = bresenham_line(m_x, m_y, n_x, n_y)
            #print(points)
            
            intensities = [img[m_y, m_x], img[n_y, n_x]] + [img[p[1], p[0]] for p in points]
            
            pred_intensities = [img[m_y, m_x], img[n_y, n_x]] + [m_I + (n_I-m_I)*dist((m_y, m_x), p)/L for p in points]
            assert len(intensities) == len(pred_intensities)

            mean_I_deltas = sum([np.abs(pred_intensities[i] - intensities[i]) for i in range(len(intensities))])/len(intensities)
            # print(L)
            collected_means.append((L, mean_I_deltas))
            #print(m, n, m_y, m_x, n_y, n_x)

    # now = datetime.datetime.now()

    # print(now-then)

    running = defaultdict(lambda: [0, 0])
    # for i, j in [(5,4),(5,2),(5,6),(6,7),(8,10),(8,20)]:
    #     running[i][0] += j
    #     running[i][1] += 1

    # averaged = [(x, running[x][0]/running[x][1]) for x in running.keys()]

    for l, i in collected_means:
        running[l][0] += i
        running[l][1] += 1

    averaged = [(x, running[x][0]/running[x][1]) for x in running.keys()]

    # print(averaged)

    L_vals_plot = [x for x, y in averaged]
    v_vals_plot = [y for x, y in averaged]
    norm_Ls = L_vals_plot/max(L_vals_plot)
    # print(L_vals_plot)
    # print(max(L_vals_plot))
    depth_norm_vs = [v/256 for v in v_vals_plot]

    print(np.trapz(v_vals_plot, norm_Ls))
    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img, cmap='gray')
    axs[0].axis('off')
    axs[1].scatter(norm_Ls, v_vals_plot)
    axs[1].set_ylim(0, 100)
    plt.savefig(f"not_normed_{name}")

# calculate_zeta(img_array, 0.025)
for image in image_list:
    img = Image.open(image).convert('L')  # Convert to grayscale
    img_loaded = np.array(img)
    print(image)
    calculate_zeta(img_loaded, 0.05, image)

print("done")