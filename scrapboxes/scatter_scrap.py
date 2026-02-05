import matplotlib.pyplot as plt


import clumpfinder
import src.data

import tqdm

# area_list = []
# axis_list = []
area_dict = {}
axis_dict = {}

colors =     ["gray", "gray", "gray", "gray", "gray", "gray", "gray", "gray", "orange", "orange", "orange", "orange", "orange", "blue", "blue", "green", "green", "blue", "blue", "blue"]
colors_ids = [   "1",    "2",    "3",    "4",    "5",    "6",  "E01",  "E02",    "E03",    "E04",    "E05",    "E06",    "E07",  "E08",  "E09",   "E10",   "E11",  "E12",  "E13",  "E14"]

f1_M = plt.figure(1)
ax1 = f1_M.add_subplot(111)
f2_m = plt.figure(2)
ax2 = f2_m.add_subplot(111)
f3_both = plt.figure(3)
ax3 = f3_both.add_subplot(111)
f4_eccen = plt.figure(4)
ax4 = f4_eccen.add_subplot(111)

for i in tqdm.tqdm(range(1, 7)):
    print(i)
    image = src.data.load_image(f"SCD_training_data/source_images/ANNOTATION/cot{i}_STOMATA_MASKS.tiff", "L")
    data_piece = clumpfinder.find_clumps_skimage(image[0], closing_threshold = 80, opening_threshold = 120, properties=("area", "axis_major_length", 'axis_minor_length', 'perimeter', 'eccentricity'))
    # area_list = [y for x in [area_list, data_piece['area']] for y in x]
    # axis_list = [y for x in [axis_list, data_piece['axis_major_length']] for y in x]
    ax1.scatter(data_piece['area'], data_piece['axis_major_length'], color = colors[i], alpha=0.05)

    ax2.scatter(data_piece['area'], data_piece['axis_minor_length'], color = colors[i], alpha=0.05)
    
    ax3.scatter(data_piece['area'], data_piece['axis_major_length'], color = colors[i], alpha=0.05)
    ax3.scatter(data_piece['area'], data_piece['axis_minor_length'], color = colors[i], alpha=0.05)
    
    ax4.scatter(data_piece['perimeter'], data_piece['eccentricity'], color = colors[i], alpha=0.05)
    # area_dict[f"{i}"] = data_piece['area']
    # axis_dict[f"{i}"] = data_piece['axis_major_length']
    # plt.savefig(f"reference_figures/cot{i}_violin.png")
    # plt.show()
for i in tqdm.tqdm(range(1, 15)):
    print(i)
    image = src.data.load_image(f"SCD_training_data/source_images/ANNOTATION/cotE{i:02d}_STOMATA_MASKS.tiff", "L")
    data_piece = clumpfinder.find_clumps_skimage(image[0], closing_threshold = 80, opening_threshold = 120, properties = ("area", "axis_major_length", 'axis_minor_length', 'perimeter', 'eccentricity'))
    # area_list = [y for x in [area_list, data_piece['area']] for y in x]
    # axis_list = [y for x in [axis_list, data_piece['axis_major_length']] for y in x]
    ax1.scatter(data_piece['area'], data_piece['axis_major_length'], color = colors[i], alpha=0.05)
    
    ax2.scatter(data_piece['area'], data_piece['axis_minor_length'], color = colors[i], alpha=0.05)
    
    ax3.scatter(data_piece['area'], data_piece['axis_major_length'], color = colors[i], alpha=0.05)
    ax3.scatter(data_piece['area'], data_piece['axis_minor_length'], color = colors[i], alpha=0.05)

    ax4.scatter(data_piece['perimeter'], data_piece['eccentricity'], color = colors[i], alpha=0.05)
    # area_dict[f"E{i:02d}"] = data_piece['area']
    # axis_dict[f"E{i:02d}"] = data_piece['axis_major_length']
    # plt.savefig(f"reference_figures/cotE{i:02d}_violin.png")
    # plt.show()


# area_list = [y for x in [area_dict[key] for key in area_dict.keys()] for y in x]
# axis_list = [y for x in [axis_dict[key] for key in axis_dict.keys()] for y in x]

# for i in range(len(colors_ids)):
#     j = colors_ids[i]
#     plt.scatter(area_dict[j], axis_dict[j], facecolors = 'none', edgecolors = colors[i], alpha=0.05)

#plt.scatter(area_list, axis_list, c=colors)
plt.show()