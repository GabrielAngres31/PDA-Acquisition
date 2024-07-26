import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import skimage
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import PIL
import numpy as np

# image = data.coins()[50:-50, 50:-50]
# print(image)
#image = skimage.io.imread("SCD_training_data/source_images/stupidtestimage_for_idiots.png", as_gray=True)
# image_inf = skimage.io.imread("inference/cot6.tif.output.png", as_gray=True)
# image_tru = skimage.io.imread("SCD_training_data/source_images/ANNOTATION/cot6_STOMATA_MASKS.tiff", as_gray=True)
image_inf = skimage.io.imread("inference/trm678_5_COT_02_SUM_trm678_ML1pmCherry-RCI2A_BRXL2pBRXL2-YFP_5dpg_022322_lif---7_fused_abaxial_merged_rotated-0002.tif.output.png", as_gray=True)
image_tru = skimage.io.imread("SCD_training_data/source_images/ANNOTATION/trm678_5_COT_02.tiff", as_gray=True)
# print(image)

# # apply threshold
# thresh = threshold_otsu(image)
# bw = closing(image > thresh, square(3))

# # remove artifacts connected to image border
# cleared = clear_border(bw)

# # label image regions
# label_image = label(cleared)
# # to make the background transparent, pass the value of `bg_label`,
# # and leave `bg_color` as `None` and `kind` as `overlay`
# image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

# skimage.io.imsave("inference/test_skimage_out.png", label_image.astype('uint8'))



# # image = data.coins()[50:-50, 50:-50]
# # print(image)
# #image = skimage.io.imread("SCD_training_data/source_images/stupidtestimage_for_idiots.png", as_gray=True)
# image_inf = skimage.io.imread("inference/cot6.tif.output.png", as_gray=True)
# image_tru = skimage.io.imread("SCD_training_data/source_images/ANNOTATION/cot6_STOMATA_MASKS.tiff", as_gray=True)


# apply threshold
thresh_inf = threshold_otsu(image_inf)
bw_inf = closing(image_inf > thresh_inf, square(3))
thresh_tru = threshold_otsu(image_tru)
bw_tru = closing(image_tru > thresh_tru, square(3))

#skimage.io.imsave("inference/test_skimage_out_otsu.png", thresh_inf.astype('uint8'))
skimage.io.imsave("inference/test_skimage_out_close.png", bw_inf.astype('uint8'))
# remove artifacts connected to image border
#print(bw_inf)
#print(bw_tru)

cleared_inf = clear_border(bw_inf)
cleared_tru = clear_border(bw_tru)

# label image regions
label_image_inf = label(cleared_inf)
label_image_tru = label(cleared_tru)

skimage.io.imsave("inference/test_skimage_out_cont.png", label_image_inf.astype('uint8'))
skimage.io.imsave("inference/test_skimage_out_cont_tru.png", label_image_tru.astype('uint8'))

#print(label_image_tru, label_image_inf)

#print([thing for thing in skimage.metrics.contingency_table(label_image_tru, label_image_inf)])
#print(skimage.metrics.contingency_table(label_image_tru, label_image_inf).shape)

#matrix_length = cont.shape[0]

# cont = skimage.metrics.contingency_table(label_image_tru, label_image_inf)

# intersections = cont.sum(axis=None)
# annotsums = cont.sum(axis=0)
# infersums = cont.sum(axis=1)
# unions =  annotsums + infersums - intersections

# IoU = intersections/unions

# IoU_cleaned = IoU
# IoU_cleaned[IoU<0] = 0

# print([i for i in cont.sum(0)])
# print(IoU_cleaned)

intersections = skimage.metrics.contingency_table(label_image_tru, label_image_inf)             #[N,M]
pixelsums_annotation = intersections.sum(axis=1)    #[N,1]
pixelsums_outputs    = intersections.sum(axis=0)    #[1,M]
unions = pixelsums_annotation + pixelsums_outputs - intersections  #[N,M]

IoU = intersections/unions
#print(IoU)

IoU_csr = IoU.tocsr()

default_list = IoU_csr[IoU_csr > 0.001]

print(str(default_list))

#IoU_csr[:, 0] = 0
#IoU_csr[0, :] = 0

#print(IoU_csr)

bins = np.arange(0, 1, 0.05) # fixed bin size

data = [x for x in default_list.tolist()[0]]
print(f"Mean: {np.mean(data)}")

#plt.xlim([min(data)-0.2, max(data)+0.2])

plt.hist(data, bins=bins, alpha=0.5)
plt.title('IoU data - mutant')
plt.xlabel('IoU (bin size = 0.05)')
plt.ylabel('Count')
plt.ylim(0, 180)
plt.savefig("inference/trm678_IoU.png")
plt.show()

#IoU_reconv = IoU_csr.tocoo()
#print(IoU_reconv)

#IoU = similarity_matrix = intersections / unions

#[print(str(cont[x][0]) + "\nfun") for x in range(matrix_length)]

# fig, ax = plt.subplots(figsize=(10, 6))
# ax.imshow(image_label_overlay)

# for region in regionprops(label_image):
#     # take regions with large enough areas
#     if region.area >= 100:
#         # draw rectangle around segmented coins
#         minr, minc, maxr, maxc = region.bbox
#         rect = mpatches.Rectangle(
#             (minc, minr),
#             maxc - minc,
#             maxr - minr,
#             fill=False,
#             edgecolor='red',
#             linewidth=2,
#         )
#         ax.add_patch(rect)

# ax.set_axis_off()
# plt.tight_layout()
# plt.show()

#print(cont[0, 5])
if False:
    import scipy.sparse as sps
    #plt.spy(cont[1:, 1:], markersize = None)
    plt.spy(IoU_cleaned, markersize = None)
    # plt.imshow(IoU[1:, 1:])
    # plt.imshow(cont[1:, 1:])
    plt.show()
    # for prec in range(0, 400, 20):
        # plt.spy(cont, precision = prec, markersize = 3)
        # plt.savefig(f"inference/cont_{prec}.png")
        # plt.clf()

#print(cont.shape)
#miss_gt = [i for i in range(cont.shape[0]) if not cont[i, 0]]
#miss_in = [i for i in range(cont.shape[1]) if not cont[0, i]]

# print(miss_gt)
# print(miss_in)
#assert miss_gt in miss_in