from random import randint

import matplotlib.pyplot as plt
from PIL import Image

from tifffile import imread, imwrite

import os
CWD = os.getcwd()
TEMP_DIR = os.path.join(CWD, "SCD training data/temp_files")
print(TEMP_DIR)
import numpy as np

from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk

import copy

import time

###--------------
#
# GUI SETUP
#
###--------------


window = Tk()
window.title("Stomata Image Classifier")
window.geometry("300x400")
frame = Frame(window)
frame.pack()

# label1.place(x=300, y=100)

window.resizable(width=False, height=False)

# FRAME CONSTRUCTION

coreframe = Frame(window)
coreframe.pack()

FRAME_imageComparison = Frame(coreframe, bg = "grey")
FRAME_imageComparison.pack(side = TOP, fill = X, expand = False)

FRAME_actionButtons = Frame(coreframe, bg = "yellow")
FRAME_actionButtons.pack(side = BOTTOM, fill = X, expand = False, pady = 12)

SUBFRAME_baseImage = Frame(FRAME_imageComparison, bg = "blue")
SUBFRAME_baseImage.pack(side = LEFT, fill = X, expand = False)

CANVAS_BASE = Canvas(SUBFRAME_baseImage, width = 104, height = 104, background = "black")
CANVAS_BASE.pack(side=TOP)

CANVAS_BASE.create_image(0, 0, anchor=NW, image=ImageTk.PhotoImage(Image.open(os.path.join(TEMP_DIR, "IMG_1058.jpg"))))

SUBFRAME_annoImage = Frame(FRAME_imageComparison, bg = "orange")
SUBFRAME_annoImage.pack(side = RIGHT, fill = X, expand = False)

CANVAS_ANNO = Canvas(SUBFRAME_annoImage, width = 104, height = 104, background = "black")
CANVAS_ANNO.pack(side=TOP)

CANVAS_ANNO.create_image(0, 0, anchor=NW, image=ImageTk.PhotoImage(Image.open(os.path.join(TEMP_DIR, "8501_GMI.jpg"))))

SUBFRAME_generatorButtons = Frame(FRAME_actionButtons)
SUBFRAME_generatorButtons.pack(side = LEFT, fill = BOTH, expand = False)

SUBFRAME_decisionsButtons = Frame(FRAME_actionButtons)
SUBFRAME_decisionsButtons.pack(side = RIGHT, fill = BOTH, expand = False)

# Button that opens file browser and allows you to select two images to use for selection

BUTTON_selectBase = Button(SUBFRAME_baseImage, text = "Open Base Image", fg = "black")#, command = lambda: IMAGE_OBJECT.setSourceImage("BASE"))
BUTTON_selectBase.pack(side=BOTTOM, fill = X)

BUTTON_selectAnno = Button(SUBFRAME_annoImage, text = "Open Annotation", fg = "black")#, command = lambda: IMAGE_OBJECT.setSourceImage("MASK"))
BUTTON_selectAnno.pack(side=BOTTOM, fill = X)

# Make Image generation button/s
print(CWD)
base_img = Image.open(os.path.join(TEMP_DIR, "IMG_1058.jpg"))
anno_img = Image.open(os.path.join(TEMP_DIR, "IMG_1058.jpg"))
# Default values ^^ are a picture of my cat

base_box = ImageTk.PhotoImage(base_img)
anno_box = ImageTk.PhotoImage(anno_img)

def reloadPreviews(img):
  global CANVAS_BASE
  global CANVAS_ANNO
  #global label_base
  #global label_anno
  global window

  global base_img
  global anno_img
  global base_box
  global anno_box

  #CANVAS_BASE.delete()
  #CANVAS_BASE.update_idletasks()
  #CANVAS_ANNO.delete()
  #CANVAS_ANNO.update_idletasks()

  CANVAS_BASE.delete('all')
  CANVAS_ANNO.delete('all')
  #CANVAS_BASE.create_image(0, 0, anchor=NW, image=base_box)
  CANVAS_BASE.create_line(0, 0, 60, 40, fill="blue")
  base_inst = CANVAS_BASE.create_image(1500, 1500, anchor=NW, image=None)#ImageTk.PhotoImage(Image.fromarray(img.BASE)))
  
  #CANVAS_BASE.update_idletasks()
  #label_base.config(image=base_box)
  
  #CANVAS_ANNO.create_image(0, 0, anchor=NW, image=anno_box)
  #anno_inst = CANVAS_ANNO.create_image(-img.cornerInit[0], -img.cornerInit[1], anchor=NW, image=ImageTk.PhotoImage(Image.fromarray(img.MASK)))
  anno_inst = CANVAS_ANNO.create_image(0, 0, anchor=NW, image=None)#ImageTk.PhotoImage(Image.fromarray(img.MASK)))
  
  #CANVAS_ANNO.update_idletasks()
  #label_anno.config(image=anno_box)
  #CANVAS_BASE.update_idletasks()


### Unbiased Random
BUTTON_unbiasedRandom = Button(SUBFRAME_generatorButtons, text = "Random Image", fg = "black")#,
                               #command = lambda: [IMAGE_OBJECT.chunkSearcher("Unbiased Random"),
                                                  #IMAGE_OBJECT.imagePrepare(),
                                                  #reloadPreviews(IMAGE_OBJECT)])
BUTTON_unbiasedRandom.pack(side=TOP, fill = X)

### Filtered Random (avoids totally blank spaces)
BUTTON_filteredRandom = Button(SUBFRAME_generatorButtons, text = "Useful Region", fg = "black")#,
                               #command = lambda: [IMAGE_OBJECT.chunkSearcher("Filtered Random"),
                               #                   IMAGE_OBJECT.imagePrepare(),
                               #                   reloadPreviews(IMAGE_OBJECT)])
BUTTON_filteredRandom.pack(side=TOP, fill = X)

### Filtered Target (Looks for sections with annotations)
BUTTON_filteredTarget = Button(SUBFRAME_generatorButtons, text = "Annotated Section", fg = "black")#,
                               #command = lambda: [IMAGE_OBJECT.chunkSearcher("Filtered Target"),
                               #                   IMAGE_OBJECT.imagePrepare(),
                               #                   reloadPreviews(IMAGE_OBJECT)])
BUTTON_filteredTarget.pack(side=TOP, fill = X)


# On Image Generation, provide buttons for classification
BUTTON_presentWithPore = Button(SUBFRAME_decisionsButtons, text = "Stomata\n(VISIBLE Pore)", fg = "black")#,
                                #command = lambda: IMAGE_OBJECT.saveClassifiedImage(IMAGE_OBJECT.BASEfilename.split("/")[-1][:-4], "stomata_pore"))
BUTTON_presentWithPore.pack(side=TOP, fill = X)

BUTTON_presentWithoutPore = Button(SUBFRAME_decisionsButtons, text = "Stomata\nNOT Visible Pore", fg = "black")#,
                                #command = lambda: IMAGE_OBJECT.saveClassifiedImage(IMAGE_OBJECT.BASEfilename.split("/")[-1][:-4], "stomata_no_pore"))
BUTTON_presentWithoutPore.pack(side=TOP, fill = X)

BUTTON_presentPartial = Button(SUBFRAME_decisionsButtons, text = "PARTIAL Stomata", fg = "black")#,
                                #command = lambda: IMAGE_OBJECT.saveClassifiedImage(IMAGE_OBJECT.BASEfilename.split("/")[-1][:-4], "stomata_partial"))
BUTTON_presentPartial.pack(side=TOP, fill = X)

BUTTON_notPresent = Button(SUBFRAME_decisionsButtons, text = "NOT PRESENT", fg = "black")#,
                                #command = lambda: IMAGE_OBJECT.saveClassifiedImage(IMAGE_OBJECT.BASEfilename.split("/")[-1][:-4], "not_present"))
BUTTON_notPresent.pack(side=TOP, fill = X)


window.mainloop()
