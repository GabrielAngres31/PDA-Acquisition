from random import randint
from PIL import Image
from tifffile import imread, imwrite
import os
#CWD = os.getcwd()
#TEMP_DIR = os.path.join(CWD, "temp_files")
import numpy as np
from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import copy
import time

DIR_CWD = os.path.join(os.getcwd(), "SCD_training_data")
DIR_TEMP = os.path.join(DIR_CWD, "temp_images")
DIR_SOURCE = os.path.join(DIR_CWD, "source_images")
DIR_BASE = os.path.join(DIR_SOURCE, "BASE")
DIR_ANNO = os.path.join(DIR_SOURCE, "ANNOTATION")

Image_BASE = None
Image_ANNO = None

Image_BASE_Tiff = None
Image_ANNO_Tiff = None

Image_BASE_size = None
Image_ANNO_size = None


window = Tk()
window.title("Stomata Image Classifier")
window.geometry("300x400")

window.resizable(width=False, height=False)

# FRAME CONSTRUCTION

buffers_dimension = 20
segment_dimension = 64
squares_dimension = 2 * buffers_dimension + segment_dimension

x_pos_0, y_pos_0, x_pos_1, y_pos_1 = 0, 0, 0, 0

main_frame = Frame(window)
main_frame.pack()

FRAME_imageComparison = Frame(main_frame, bg = "grey")
FRAME_imageComparison.pack(side = TOP, fill = X, expand = False)

FRAME_actionButtons = Frame(main_frame, bg = "yellow")
FRAME_actionButtons.pack(side = BOTTOM, fill = X, expand = False, pady = 12)

FRAME_baseImage = Frame(FRAME_imageComparison, bg = "blue")
FRAME_baseImage.pack(side = LEFT, fill = X, expand = False)

CANVAS_BASE = Canvas(FRAME_baseImage, width = squares_dimension, height = squares_dimension, background = "black")
CANVAS_BASE.pack(side=TOP)

FRAME_annoImage = Frame(FRAME_imageComparison, bg = "orange")
FRAME_annoImage.pack(side = RIGHT, fill = X, expand = False)

CANVAS_ANNO = Canvas(FRAME_annoImage, width = squares_dimension, height = squares_dimension, background = "black")
CANVAS_ANNO.pack(side=TOP)

FRAME_generateButtons = Frame(FRAME_actionButtons)
FRAME_generateButtons.pack(side = LEFT, fill = BOTH, expand = False)

FRAME_decideButtons = Frame(FRAME_actionButtons)
FRAME_decideButtons.pack(side = RIGHT, fill = BOTH, expand = False)

# Button that opens file browser and allows you to select two images to use for selection

BUTTON_selectBase = Button(FRAME_baseImage, text = "Open Base Image", fg = "black", command = None)
BUTTON_selectBase.pack(side=BOTTOM, fill = X)

BUTTON_selectAnno = Button(FRAME_annoImage, text = "Open Annotation", fg = "black", command = None)
BUTTON_selectAnno.pack(side=BOTTOM, fill = X)


### Unbiased Random
BUTTON_generateUnbiasedRandom = Button(FRAME_generateButtons, text = "Random Image", fg = "black",
                               command = lambda: [
                                   getValidBoxCoords("Unbiased Random"),
                                   update_canvases(-x_pos_0, -y_pos_0)])
BUTTON_generateUnbiasedRandom.pack(side=TOP, fill = X)

### Filtered Random (avoids totally blank spaces)
BUTTON_generateFilteredRandom = Button(FRAME_generateButtons, text = "Useful Region", fg = "black",
                               command = lambda: [
                                   getValidBoxCoords("Filtered Random"),
                                   update_canvases(-x_pos_0, -y_pos_0)])
BUTTON_generateFilteredRandom.pack(side=TOP, fill = X)

### Filtered Target (Looks for sections with annotations)
BUTTON_generateFilteredTarget = Button(FRAME_generateButtons, text = "Annotated Section", fg = "black",
                               command = lambda: [
                                   getValidBoxCoords("Filtered Target"),
                                   update_canvases(-x_pos_0, -y_pos_0)])
BUTTON_generateFilteredTarget.pack(side=TOP, fill = X)

# BUTTON_generateEnclosedTarget = Button(FRAME_generateButtons, text = "Enclosed Annotation", fg = "black",
#                                command = lambda: [
#                                    getValidBoxCoords("Filtered Target"),
#                                    update_canvases(-x_pos_0, -y_pos_0)])
# BUTTON_generateEnclosedTarget.pack(side=TOP, fill = X)


# On Image Generation, provide buttons for classification
BUTTON_decidePresentWithPore = Button(FRAME_decideButtons, text = "Stomata\n(VISIBLE Pore)", fg = "black",
                                command = None)
BUTTON_decidePresentWithPore.pack(side=TOP, fill = X)

BUTTON_decidePresentWithoutPore = Button(FRAME_decideButtons, text = "Stomata\nNOT Visible Pore", fg = "black",
                                command = None)
BUTTON_decidePresentWithoutPore.pack(side=TOP, fill = X)

BUTTON_decidePresentPartial = Button(FRAME_decideButtons, text = "PARTIAL Stomata", fg = "black",
                                command = None)
BUTTON_decidePresentPartial.pack(side=TOP, fill = X)

BUTTON_decideNotPresent = Button(FRAME_decideButtons, text = "NOT PRESENT", fg = "black",
                                command = None)
BUTTON_decideNotPresent.pack(side=TOP, fill = X)


image_base_filepath = None
image_anno_filepath = None

def setImages():
    global image_base_filepath
    image_base_filepath = fd.askopenfilename(title = "Select a BASE Image", initialdir = DIR_BASE)

    global Image_BASE
    Image_BASE = ImageTk.PhotoImage(Image.open(image_base_filepath))

    global Image_BASE_size
    Image_BASE_size = [Image_BASE.width(), Image_BASE.height()]
    print(image_base_filepath)
    print(Image_BASE_size)

    global Image_BASE_Tiff
    Image_BASE_Tiff = imread(image_base_filepath)

    global image_anno_filepath
    image_anno_filepath = fd.askopenfilename(title = "Select a ANNO Image", initialdir = DIR_ANNO)

    global Image_ANNO
    Image_ANNO = ImageTk.PhotoImage(Image.open(image_anno_filepath))

    global Image_ANNO_size
    Image_ANNO_size = [Image_ANNO.width(), Image_ANNO.height()]
    print(image_anno_filepath)
    print(Image_ANNO_size)

    global Image_ANNO_Tiff
    Image_ANNO_Tiff = imread(image_anno_filepath)

setImages()

def getValidBoxCoords(choice="Unbiased Random"):
    global Image_BASE_size
    global Image_ANNO_size

    global segment_dimension
    global buffers_dimension

    global x_pos_0
    global x_pos_1
    global y_pos_0
    global y_pos_1

    global Image_BASE_Tiff
    global Image_ANNO_Tiff

    # TODO: Put global chunklet variables here
    # TODO: Finalize the saving and storing buttons

    assert Image_BASE_size[0] == Image_ANNO_size[0]
    assert Image_BASE_size[1] == Image_ANNO_size[1]

    width  = Image_BASE_size[0]
    height = Image_BASE_size[1]

    x0, y0, x1, y1 = 0, 0, 0, 0

    isQualifiedChunk = False
    while not isQualifiedChunk:
        x0 = randint(0, width  - segment_dimension - 1)
        y0 = randint(0, height - segment_dimension - 1)

        x1 = x0 + segment_dimension - 1
        y1 = y0 + segment_dimension - 1 

        if choice == "Unbiased Random":
            isQualifiedChunk = True
        elif choice == "Filtered Random":
            basechunk = Image_BASE_Tiff[(y0-buffers_dimension):(y1+buffers_dimension+1), 
                                        (x0-buffers_dimension):(x1+buffers_dimension+1)]
            base_image_mean = basechunk.mean()
            if base_image_mean > 25: # THIS VALUE WAS OBTAINED BY MANUAL TESTING
                isQualifiedChunk = True
        elif choice == "Filtered Target":
            annochunk = Image_ANNO_Tiff[(y0+buffers_dimension):(y1-buffers_dimension+1), 
                                        (x0+buffers_dimension):(x1-buffers_dimension+1)]
            anno_image_mean = annochunk.mean()
            if anno_image_mean > 15:
                isQualifiedChunk = True
        else:
            raise Exception("Invalid Chunktype")

    x_pos_0, y_pos_0, x_pos_1, y_pos_1 = x0, y0, x1, y1
    print(x_pos_0, y_pos_0)

def drawBox(canvas_obj, size, buffer, color="blue"):

    canvas_obj.create_line(buffer,      buffer,      buffer+size, buffer,      fill = color)
    canvas_obj.create_line(buffer,      buffer,      buffer,      buffer+size, fill = color)
    canvas_obj.create_line(buffer+size, buffer,      buffer+size, buffer+size, fill = color)
    canvas_obj.create_line(buffer,      buffer+size, buffer+size, buffer+size, fill = color)

def update_canvases(x_pos, y_pos):
    CANVAS_BASE.delete()
    CANVAS_ANNO.delete()

    CANVAS_BASE.create_image(x_pos, y_pos, anchor = NW, image = Image_BASE)
    CANVAS_ANNO.create_image(x_pos, y_pos, anchor = NW, image = Image_ANNO)

    drawBox(CANVAS_BASE, segment_dimension, buffers_dimension)
    drawBox(CANVAS_ANNO, segment_dimension, buffers_dimension)

mainloop()