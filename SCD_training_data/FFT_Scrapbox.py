import numpy as np
import matplotlib.pyplot as plt
import os

from skimage.io import imread, imshow, imsave

img_list = ["COT1_719y-1687x", "COT1_719y-1688x", "COT1_719y-1689x", "COT1_719y-1690x", "COT1_719y-1691x", "COT1_719y-1692x", "COT1_719y-1693x"]
IMG_DIR = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\generated\\test\\base\\COT1_1081y-2133x"

# img_files = [imread(os.path.join(IMG_DIR, f"{img}.png")) for img in img_list]
# fft_files = [np.log(abs(np.fft.fftshift(np.fft.fft2(img)))) for img in img_files]

# transforms_zip = zip(img_files, fft_files)



# img = imread("C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\generated\\temp_base.jpg")

# plt.figure(num=None, figsize=(8, 8), dpi=80)
# plt.imshow(img, cmap='gray')


# fft_img = np.fft.fftshift(np.fft.fft2(img))
# plt.figure(num=None, figsize=(2, 2), dpi=80)
# plt.imshow(np.log(abs(fft_img)), cmap='gray')

# plt.show()



# plt.figure(1)
# f = len(img_files)
# for i,imgs in enumerate(transforms_zip):

#     plot_objects = (imgs[0], imgs[1])


#     p = len(plot_objects)
#     for j,plot_obj in enumerate(plot_objects):
#         plt.subplot(p, f, i+f*j+1)
#         plt.imshow(plot_obj, cmap='gray') #, vmin = 0, vmax = 255)

# plt.show()


def fft_avg_investigator(dir):
    imgs = [imread(os.path.join(dir, img)) for img in os.listdir(dir)]
    fft_imgs = [np.log(abs(np.fft.fftshift(np.fft.fft2(img)))) for img in imgs]
    
    return np.mean( np.array(fft_imgs), axis=0 )

# fft_avg_out = fft_avg_investigator(IMG_DIR)

# plt.figure(num=None, figsize=(6, 6), dpi=80)
# plt.imshow(np.log(abs(fft_avg_out)), cmap='gray')
# plt.show()

def fft_bulk_return(dir, func = lambda x: np.log(abs(np.fft.fftshift(np.fft.fft2(x))))):
    imgs = [imread(os.path.join(dir, img)) for img in os.listdir(dir)]
    fft_imgs = [func(img) for img in imgs]
    
    return fft_imgs

def symmetry_remap(arr, func = lambda x: 255 - np.abs(x - np.flip(x))):
    assert arr.shape[0] == arr.shape[1]
    np.clip(arr, 0, 255, out = arr)
    assert np.all(arr >= 0) and np.all(arr <= 255), f"{arr}"
    parity = arr.size % 2
    center = arr.shape[0]//2
    arr_out = np.full(arr.shape, 0)

    #print(f"Parity _{parity}_ || Center _{center}_")
    
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

FFT_OUT_DIR = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\generated\\test\\FFT_TEST"

for i, img in enumerate(fft_bulk_return(IMG_DIR)):
    imsave(os.path.join(FFT_OUT_DIR, f"file_{i}.png"), (img*256).astype("uint8"))

# for img in [imread(os.path.join(FFT_OUT_DIR, file)) for file in os.listdir(FFT_OUT_DIR)]:
#     print("doing")
#     imsave(os.path.join("C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\generated\\test\\FFT_TEST_SYMM", f"file_{i}.png"), symmetry_remap(img))

# plt.figure(1)
# plt.subplot(121)
# plt.plot(imread(os.path.join(FFT_OUT_DIR, "file_0.png")))

# plt.plot(symmetry_remap(imread(os.path.join(FFT_OUT_DIR, "file_0.png"))))

pic = imread(os.path.join(FFT_OUT_DIR, "file_0.png"))
transform = symmetry_remap(imread(os.path.join(FFT_OUT_DIR, "file_0.png")))

# plt.show()



plt.figure(1)
# plt.subplot(121)

# imshow(pic)
# plt.subplot(122)
# imshow(transform)

imshow(symmetry_remap(imread("C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\testchecker.png")))

plt.show()