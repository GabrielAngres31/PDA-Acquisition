import matplotlib.pyplot as plt
import seaborn as sns

import clumpfinder
import src.data

data_dict = {}

for i in range(1, 15):
    print(i)
    image = src.data.load_image(f"SCD_training_data/source_images/ANNOTATION/cotE{i:02d}_STOMATA_MASKS.tiff", "L")
    data_piece = clumpfinder.find_clumps_skimage(image[0], closing_threshold = 80, opening_threshold = 120)['area']
    sns.violinplot(data_piece)
    plt.savefig(f"reference_figures/cotE{i:02d}_violin.png")
    plt.show()
    data_dict[f"cotE{i:02d}"] = data_piece
for i in range(1, 7):
    print(i)
    image = src.data.load_image(f"SCD_training_data/source_images/ANNOTATION/cot{i}_STOMATA_MASKS.tiff", "L")
    data_piece = clumpfinder.find_clumps_skimage(image[0], closing_threshold = 80, opening_threshold = 120)['area']
    sns.violinplot(data_piece)
    plt.savefig(f"reference_figures/cot{i}_violin.png")
    plt.show()
    data_dict[f"cot{i}"] = data_piece

print(data_dict.keys())
sns.violinplot(data=[val for val in data_dict.values()])

plt.show()