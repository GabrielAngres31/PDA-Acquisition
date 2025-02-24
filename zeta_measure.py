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

for m_y in range(img_array.shape[0]):
    for m_x in range(img_array.shape[1]):
        m_I = img_array[m_y, m_x]
        for n_y in range(img_array.shape[0]): # m_y-1, 
            for n_x in range(img_array.shape[1]): # m_x-1, 
                if n_y == m_y and n_x == m_x:
                    continue
                if np.random.rand() < 0.9:
                    continue
                
                n_I = img_array[n_y, n_x]
                #print(m_I)
                #print(n_I)
                d_I = n_I-m_I
                def dist(p_a, p_b): return np.sqrt(np.sum(np.square(np.array(p_a) -np.array(p_b)))) 
                L = dist((n_y, n_x), (m_y, m_x))
                points = bresenham_line(m_x, m_y, n_x, n_y)
                #print(points)
                
                intensities = [img_array[m_y, m_x], img_array[n_y, n_x]] + [img_array[p[1], p[0]] for p in points]
                
                pred_intensities = [img_array[m_y, m_x], img_array[n_y, n_x]] + [m_I + (n_I-m_I)*dist((m_y, m_x), p)/L for p in points]
                assert len(intensities) == len(pred_intensities)

                mean_I_deltas = sum([np.abs(pred_intensities[i] - intensities[i]) for i in range(len(intensities))])/len(intensities)

                collected_means.append((L, mean_I_deltas))

                # print(mean_I_deltas, L)
                
                #print(f"{img_array[m_y,m_x]}, {img_array[n_y, n_x]}")

running = defaultdict(lambda: [0, 0])
# for i, j in [(5,4),(5,2),(5,6),(6,7),(8,10),(8,20)]:
#     running[i][0] += j
#     running[i][1] += 1

# averaged = [(x, running[x][0]/running[x][1]) for x in running.keys()]

for l, i in collected_means:
    running[l][0] += i
    running[l][1] += 1

averaged = [(x, running[x][0]/running[x][1]) for x in running.keys()]

print(averaged)

L_vals_plot = [x for x, y in averaged]
v_vals_plot = [y for x, y in averaged]
norm_Ls = L_vals_plot/max(L_vals_plot)

plt.scatter(norm_Ls, v_vals_plot)
plt.show()

print("done")