import matplotlib.pyplot as plt


import clumpfinder
import src.data

area_list = []
axis_list = []


for i in range(1, 15):
    print(i)
    image = src.data.load_image(f"SCD_training_data/source_images/ANNOTATION/cotE{i:02d}_STOMATA_MASKS.tiff", "L")
    data_piece = clumpfinder.find_clumps_skimage(image[0], closing_threshold = 80, opening_threshold = 120)
    area_list = [y for x in [area_list, data_piece['area']] for y in x]
    axis_list = [y for x in [axis_list, data_piece['axis_major_length']] for y in x]
    # plt.savefig(f"reference_figures/cotE{i:02d}_violin.png")
    # plt.show()
for i in range(1, 7):
    print(i)
    image = src.data.load_image(f"SCD_training_data/source_images/ANNOTATION/cot{i}_STOMATA_MASKS.tiff", "L")
    data_piece = clumpfinder.find_clumps_skimage(image[0], closing_threshold = 80, opening_threshold = 120)
    area_list = [y for x in [area_list, data_piece['area']] for y in x]
    axis_list = [y for x in [axis_list, data_piece['axis_major_length']] for y in x]
    # plt.savefig(f"reference_figures/cot{i}_violin.png")
    # plt.show()
    
plt.scatter(area_list, axis_list)
plt.show()