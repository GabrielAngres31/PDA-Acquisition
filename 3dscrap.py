import matplotlib.pyplot as plt


import clumpfinder
import src.data
import tqdm

import plotly.express as px
# area_list = []
# axis_list = []
area_dict = {}
axis_dict = {}
minx_dict = {}

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

colors_dict = {"N/A":"gray", "3":"orange", "4":"green", "5":"blue",}

colors =     ["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",   "4",   "4",   "4",   "4",   "4",   "5",   "5",   "3",   "3",   "5",   "5",   "5"]
colors_ids = [  "1",   "2",   "3",   "4",   "5",   "6", "E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09", "E10", "E11", "E12", "E13", "E14"]

def plot3dhandler(fields: tuple):
    for i in tqdm.tqdm(range(1, 7)):
        print(i)
        image = src.data.load_image(f"SCD_training_data/source_images/ANNOTATION/cot{i}_STOMATA_MASKS.tiff", "L")
        data_piece = clumpfinder.find_clumps_skimage(image[0], closing_threshold = 80, opening_threshold = 120, properties = fields)
        # area_list = [y for x in [area_list, data_piece['area']] for y in x]
        # axis_list = [y for x in [axis_list, data_piece['axis_major_length']] for y in x]
        ax.scatter(data_piece[fields[0]], data_piece[fields[1]], data_piece[fields[2]], color = colors_dict[colors[i]])
        # area_dict[f"{i}"] = data_piece['area']
        # axis_dict[f"{i}"] = data_piece['axis_major_length']
        # minx_dict[f"{i}"] = data_piece['axis_minor_length']
        # plt.savefig(f"reference_figures/cot{i}_violin.png")
        # plt.show()
    for i in tqdm.tqdm(range(1, 15)):
        print(i)
        image = src.data.load_image(f"SCD_training_data/source_images/ANNOTATION/cotE{i:02d}_STOMATA_MASKS.tiff", "L")
        data_piece = clumpfinder.find_clumps_skimage(image[0], closing_threshold = 80, opening_threshold = 120, properties = fields)
        # area_list = [y for x in [area_list, data_piece['area']] for y in x]
        # axis_list = [y for x in [axis_list, data_piece['axis_major_length']] for y in x]
        ax.scatter(data_piece[fields[0]], data_piece[fields[1]], data_piece[fields[2]], color = colors_dict[colors[i]])
        # area_dict[f"E{i:02d}"] = data_piece['area']
        # axis_dict[f"E{i:02d}"] = data_piece['axis_major_length']
        # minx_dict[f"E{i:02d}"] = data_piece['axis_minor_length']
        # plt.savefig(f"reference_figures/cotE{i:02d}_violin.png")
        # plt.show()
    plt.show()

    # area_list = [y for x in [area_dict[key] for key in area_dict.keys()] for y in x]
    # axis_list = [y for x in [axis_dict[key] for key in axis_dict.keys()] for y in x]




    # for i in range(len(colors_ids)):
    #     j = colors_ids[i]
    #     ax.scatter(area_dict[j], axis_dict[j], minx_dict[j], color = colors[i])

    #plt.scatter(area_list, axis_list, c=colors)


# plot3dhandler(['area', 'eccentricity', 'axis_minor_length'])
plot3dhandler(('area', 'perimeter', 'eccentricity'))