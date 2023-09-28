from random import randint
from PIL import Image
from tifffile import imread, imwrite
import os
import numpy as np
from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk


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

image_base_filepath = None
image_anno_filepath = None

def setImagesBase():
    global image_base_filepath
    image_base_filepath = fd.askopenfilename(title = "Select a BASE Image", initialdir = DIR_BASE, 
                                             initialfile="C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\BASE\\cot1.tif")

    global Image_BASE
    Image_BASE = ImageTk.PhotoImage(Image.open(image_base_filepath))

    global Image_BASE_size
    Image_BASE_size = [Image_BASE.width(), Image_BASE.height()]
    print(image_base_filepath)
    print(Image_BASE_size)

    global Image_BASE_Tiff
    Image_BASE_Tiff = imread(image_base_filepath)

def setImagesAnno():
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

def saveChunk(x0, y0, x1, y1, target_folder):
    print("Saving Chunk...")
    chunk = Image_BASE_Tiff[y0:y1, 
                            x0:x1]
    imwrite(f"C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\generated\\{target_folder}\\{target_folder}-{x0}x-{y0}y.jpg", chunk)

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

BUTTON_selectBase = Button(FRAME_baseImage, text = "Open Base Image", fg = "black", command = lambda: setImagesBase())
BUTTON_selectBase.pack(side=BOTTOM, fill = X)

BUTTON_selectAnno = Button(FRAME_annoImage, text = "Open Annotation", fg = "black", command = lambda: setImagesAnno())
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
                                   update_canvases(-(x_pos_0-buffers_dimension), -(y_pos_0-buffers_dimension))])
BUTTON_generateFilteredTarget.pack(side=TOP, fill = X)

BUTTON_generateEnclosedTarget = Button(FRAME_generateButtons, text = "Enclosed Annotation", fg = "black",
                                command = lambda: [
                                    getValidBoxCoords("Enclosed Target"),
                                    update_canvases(-(x_pos_0-buffers_dimension), -(y_pos_0-buffers_dimension))])
BUTTON_generateEnclosedTarget.pack(side=TOP, fill = X)


# On Image Generation, provide buttons for classification
BUTTON_decidePresentWithPore = Button(FRAME_decideButtons, text = "Stomata\n(VISIBLE Pore)", fg = "black",
                                command = lambda: saveChunk(x_pos_0, y_pos_0, x_pos_1, y_pos_1, "stomata_pore"))
BUTTON_decidePresentWithPore.pack(side=TOP, fill = X)

BUTTON_decidePresentWithoutPore = Button(FRAME_decideButtons, text = "Stomata\nNOT Visible Pore", fg = "black",
                                command = lambda: saveChunk(x_pos_0, y_pos_0, x_pos_1, y_pos_1, "stomata_without"))
BUTTON_decidePresentWithoutPore.pack(side=TOP, fill = X)

BUTTON_decidePresentPartial = Button(FRAME_decideButtons, text = "PARTIAL Stomata", fg = "black",
                                command = lambda: saveChunk(x_pos_0, y_pos_0, x_pos_1, y_pos_1, "partial_stomama"))
BUTTON_decidePresentPartial.pack(side=TOP, fill = X)

BUTTON_decideNotPresent = Button(FRAME_decideButtons, text = "NOT PRESENT", fg = "black",
                                command = lambda: saveChunk(x_pos_0, y_pos_0, x_pos_1, y_pos_1, "no_stomata"))
BUTTON_decideNotPresent.pack(side=TOP, fill = X)




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

    # TODO: Finalize the saving and storing buttons

    assert Image_BASE_size[0] == Image_ANNO_size[0]
    assert Image_BASE_size[1] == Image_ANNO_size[1]

    width  = Image_BASE_size[0]
    height = Image_BASE_size[1]

    x0, y0, x1, y1 = 0, 0, 0, 0

    isQualifiedChunk = False
    while not isQualifiedChunk:
        print("Looking")
        x0 = randint(0, width  - segment_dimension + 1)
        y0 = randint(0, height - segment_dimension + 1)

        x1 = x0 + segment_dimension - 1
        y1 = y0 + segment_dimension - 1 

        if choice == "Unbiased Random":
            print("lol")
            isQualifiedChunk = True
        elif choice == "Filtered Random":
            print("thinking")
            basechunk = Image_BASE_Tiff[(y0-buffers_dimension):(y1+buffers_dimension+1), 
                                        (x0-buffers_dimension):(x1+buffers_dimension+1)]
            base_image_mean = np.mean(basechunk)
            if base_image_mean > 25: # THIS VALUE WAS OBTAINED BY MANUAL TESTING
                isQualifiedChunk = True
        elif choice == "Filtered Target":
            
            annochunk = Image_ANNO_Tiff[(y0):(y1), 
                                        (x0):(x1)]
            # THIS SHOULDN'T WORK THE WAY IT DOES, BECAUSE IT WOULD DEFINE A 24x24 square
            # BUT setting it to y0:y1+1 and x0:x1+1 sometimes generates sections that have 
            #   annotations in the buffer area but not the inspection area.
            # IT'S WEIRD.
            #print(annochunk)
            anno_image_mean = np.mean(annochunk)
            print(anno_image_mean)
            if anno_image_mean > 0:
                isQualifiedChunk = True
        elif choice == "Enclosed Target":
            print("Welp")
            annochunk = Image_ANNO_Tiff[y0:(y1+1), 
                                        x0:(x1+1)]
            
            anno_image_mean = np.mean(annochunk)
            if anno_image_mean > 0:
                print(annochunk)
                isQualifiedChunk = True
                for n in list(range(0, segment_dimension-1)):
                    if not(isQualifiedChunk):
                        break
                    if (annochunk[0][n]) or annochunk[n][segment_dimension-1] or annochunk[segment_dimension-1][segment_dimension-n-1] or annochunk[segment_dimension-n-1][0]:
                        isQualifiedChunk = False
                        break
        else:
            raise Exception("Invalid Chunktype")

    x_pos_0, y_pos_0, x_pos_1, y_pos_1 = x0, y0, x1, y1
    #print(x_pos_0, y_pos_0)

def drawBox(canvas_obj, size, buffer, color="blue"):

    canvas_obj.create_line(buffer,      buffer,      buffer+size, buffer,      fill = color)
    canvas_obj.create_line(buffer,      buffer,      buffer,      buffer+size, fill = color)
    canvas_obj.create_line(buffer+size, buffer,      buffer+size, buffer+size, fill = color)
    canvas_obj.create_line(buffer,      buffer+size, buffer+size, buffer+size, fill = color)

def update_canvases(x_pos, y_pos):
    CANVAS_BASE.delete("all")
    CANVAS_ANNO.delete("all")

    CANVAS_BASE.create_image(x_pos, y_pos, anchor = NW, image = Image_BASE)
    CANVAS_ANNO.create_image(x_pos, y_pos, anchor = NW, image = Image_ANNO)

    drawBox(CANVAS_BASE, segment_dimension, buffers_dimension)
    drawBox(CANVAS_ANNO, segment_dimension, buffers_dimension)


mainloop()

