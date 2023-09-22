###--------------
#
# Import packages
#
###--------------

#import keras
#import tensorflow as tf

from random import randint

import matplotlib.pyplot as plt
from PIL import Image

from tifffile import imread, imwrite

#import os

import numpy as np

from tkinter import *
from PIL import Image, ImageTk

import copy

print("Imported Packages")

###--------------
#
# monoToRGB Utility Function
#
###--------------

def monoToRGB(img):
  return np.array([[[i, i, i] for i in j] for j in img])

###--------------
#
# SAMPLE_IMAGE Class
#   Handles TIFF file import and chunk return
#
###--------------

class SAMPLER_IMAGE:
  def __init__(self, BASE, MASK, segment_size, buffer_size):

    self.BASEmono = BASE

    self.BASE = np.array(monoToRGB(BASE))
    self.MASK = MASK

    self.BASE_box = copy.deepcopy(np.array(monoToRGB(BASE))) #TODO
    self.MASK_box = copy.deepcopy(MASK)

    assert len(BASE[0]) == len(MASK[0]), "Image WIDTH != "
    assert    len(BASE) == len(MASK),    "Image HEIGHT != "
    self.size = (len(self.BASE[0]), len(self.BASE))

    self.segment_size = segment_size
    self.buffer_size = buffer_size

    self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0
    self.cornerInit = [0, 0]
    self.cornerFinal= [0, 0]

    self.areaType = "BASE"
    self.areaSect = self.BASE[0:2, 0:2]

  def setGivenArea(self, choice = "ALL"):
    self.x1 = randint(0, self.size[0] - self.segment_size - 1)
    self.y1 = randint(0, self.size[1] - self.segment_size - 1)

    self.x2 = self.x1 + self.segment_size - 1
    self.y2 = self.y1 + self.segment_size - 1

    self.cornerInit = [self.x1, self.y1]
    self.cornerFinal = [self.x2, self.y2]

  def grabChunk(self, choice, buffer):
    assert choice in ["BASE_box", "MASK_box"]
    return getattr(self, choice)[(self.y1-buffer):(self.y2+buffer+1), (self.x1-buffer):(self.x2+buffer+1)]

  def drawBoxes(self, color=(0, 0, 255)):
    self.BASE_box = copy.deepcopy(self.BASE)  #TODO
    self.MASK_box = copy.deepcopy(self.MASK)  #TODO
    
    # Draw top and bottom edges
    for x in range(self.x1, self.x2+1):
      
      self.BASE_box[self.y1-1][x] = color
      self.MASK_box[self.y1-1][x] = color
      
      self.BASE_box[self.y2+1][x] = color
      self.MASK_box[self.y2+1][x] = color      

    # Draw left and right edges
    for y in range(self.y1, self.y2+1):
      self.BASE_box[y][self.x1-1] = color
      self.MASK_box[y][self.x1-1] = color
      
      self.BASE_box[y][self.x2+1] = color
      self.MASK_box[y][self.x2+1] = color 


  def chunkSearcher(self, chunkType):
    isQualifiedChunk = False
    while not isQualifiedChunk:
      self.setGivenArea()
      if chunkType == "Unbiased Random":
        isQualifiedChunk = True
      elif chunkType == "Filtered Random":
        image_mean = self.grabChunk("BASE_box", buffer = 0).mean()
        print(image_mean)
        if image_mean > 25: # THIS VALUE WAS OBTAINED BY MANUAL TESTING
          isQualifiedChunk = True
      elif chunkType == "Filtered Target":
        image_mean = self.grabChunk("MASK_box", buffer = 0).mean()
        print(image_mean)
        if image_mean > 2:
          isQualifiedChunk = True
      else:
        raise Exception("Invalid Chunktype")
    
        #image_mean_yellow = self.grabChunk(
  def imagePrepare(self):

    self.drawBoxes()
    
    imwrite(f"temp_files/temp_base.jpg", self.grabChunk("BASE_box", buffer = 0))
    imwrite(f"temp_files/temp_anno.jpg", self.grabChunk("MASK_box", buffer = 0))

    imwrite(f"temp_files/temp_base_box.jpg", self.grabChunk("BASE_box", buffer = self.buffer_size))
    imwrite(f"temp_files/temp_anno_box.jpg", self.grabChunk("MASK_box", buffer = self.buffer_size))
##
## Instantiate IMAGE_OBJECT
##

print("Instantiating image objects")
IMAGE_OBJECT = SAMPLER_IMAGE(BASE = imread("C:/Users/gjang/Documents/GitHub/PDA-Acquisition/SCD training data/120621_cot1_max_rotated_c2.tif"),
                             MASK = imread("C:/Users/gjang/Documents/GitHub/PDA-Acquisition/SCD training data/120621_cot1_max_rotated_c2_MASK.tif"),
                             segment_size = 64,
                             buffer_size = 20)

##
## drawBox for putting a square around target regions
##

##
## Run multiple functions at once for button commands
##

###--------------
#
# GUI SETUP
#
###--------------

print
window = Tk()
window.title("Stomata Image Classifier")
window.geometry("400x400")
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

CANVAS_BASE = Canvas(SUBFRAME_baseImage, width = IMAGE_OBJECT.segment_size+2*IMAGE_OBJECT.buffer_size, height = IMAGE_OBJECT.segment_size+2*IMAGE_OBJECT.buffer_size)
CANVAS_BASE.pack(side=TOP)

SUBFRAME_annoImage = Frame(FRAME_imageComparison, bg = "orange")
SUBFRAME_annoImage.pack(side = RIGHT, fill = X, expand = False)

CANVAS_ANNO = Canvas(SUBFRAME_annoImage, width = IMAGE_OBJECT.segment_size+2*IMAGE_OBJECT.buffer_size, height = IMAGE_OBJECT.segment_size+2*IMAGE_OBJECT.buffer_size)
CANVAS_ANNO.pack(side=TOP)

SUBFRAME_generatorButtons = Frame(FRAME_actionButtons)
SUBFRAME_generatorButtons.pack(side = LEFT, fill = BOTH, expand = False)

SUBFRAME_decisionsButtons = Frame(FRAME_actionButtons)
SUBFRAME_decisionsButtons.pack(side = RIGHT, fill = BOTH, expand = False)

# Button that opens file browser and allows you to select two images to use for selection

BUTTON_selectBase = Button(SUBFRAME_baseImage, text = "Open Base Image", fg = "black")
BUTTON_selectBase.pack(side=BOTTOM, fill = X)

BUTTON_selectAnno = Button(SUBFRAME_annoImage, text = "Open Annotation", fg = "black")
BUTTON_selectAnno.pack(side=BOTTOM, fill = X)

# Make Image generation button/s

def reloadImages():
  global CANVAS_BASE
  global CANVAS_ANNO
  #global label_base
  #global label_anno
  global window

  #CANVAS_BASE.delete()
  #CANVAS_BASE.update_idletasks()
  #CANVAS_ANNO.delete()
  #CANVAS_ANNO.update_idletasks()

  base_img = Image.open("C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD training data\\temp_files\\temp_base_box.jpg")
  base_box = ImageTk.PhotoImage(base_img)
  #CANVAS_BASE.create_image(0, 0, anchor=NW, image=base_box)
  base_inst = CANVAS_BASE.create_image(0, 0, anchor=NW, image=base_box)
  print(base_inst)
  #CANVAS_BASE.update_idletasks()
  #label_base.config(image=base_box)

  anno_img = Image.open("C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD training data\\temp_files\\temp_anno_box.jpg")
  anno_box = ImageTk.PhotoImage(anno_img)
  print(type(anno_box))
  #CANVAS_ANNO.create_image(0, 0, anchor=NW, image=anno_box)
  anno_inst = CANVAS_ANNO.create_image(0, 0, anchor=NW, image=anno_box)
  
  print(anno_inst)
  #CANVAS_ANNO.update_idletasks()
  #label_anno.config(image=anno_box)
  window.update_idletasks()


### Unbiased Random
BUTTON_unbiasedRandom = Button(SUBFRAME_generatorButtons, text = "Random Image", fg = "black",
                               command = lambda: [IMAGE_OBJECT.chunkSearcher("Unbiased Random"),
                                                  IMAGE_OBJECT.imagePrepare(),
                                                  reloadImages(),
                                                  print("RI")])
BUTTON_unbiasedRandom.pack(side=TOP, fill = X)

### Filtered Random (avoids totally blank spaces)
BUTTON_filteredRandom = Button(SUBFRAME_generatorButtons, text = "Useful Region", fg = "black",
                               command = lambda: [IMAGE_OBJECT.chunkSearcher("Filtered Random"),
                                                  IMAGE_OBJECT.imagePrepare(),
                                                  reloadImages(),
                                                  print("UR")])
BUTTON_filteredRandom.pack(side=TOP, fill = X)

### Filtered Target (Looks for sections with annotations)
BUTTON_filteredTarget = Button(SUBFRAME_generatorButtons, text = "Annotated Section", fg = "black",
                               command = lambda: [IMAGE_OBJECT.chunkSearcher("Filtered Target"),
                                                  IMAGE_OBJECT.imagePrepare(),
                                                  reloadImages(),
                                                  print("AS:")])
BUTTON_filteredTarget.pack(side=TOP, fill = X)


# On Image Generation, provide buttons for classification
BUTTON_presentWithPore = Button(SUBFRAME_decisionsButtons, text = "Stomata\n(VISIBLE Pore)", fg = "black")
BUTTON_presentWithPore.pack(side=TOP, fill = X)

BUTTON_presentWithoutPore = Button(SUBFRAME_decisionsButtons, text = "Stomata\nNOT Visible Pore", fg = "black")
BUTTON_presentWithoutPore.pack(side=TOP, fill = X)

BUTTON_notPresent = Button(SUBFRAME_decisionsButtons, text = "NOT PRESENT", fg = "black")
BUTTON_notPresent.pack(side=TOP, fill = X)

BUTTON_unclear = Button(SUBFRAME_decisionsButtons, text = "UNCLEAR", fg = "black")
BUTTON_unclear.pack(side=TOP, fill = X)

# Image space initialization

#base_img = None
#anno_img = None

#label_base = Label(SUBFRAME_baseImage, image=base_img)

#label_base.pack(side=TOP)

#label_anno = Label(SUBFRAME_annoImage, image=anno_img)

#label_anno.pack(side=TOP)

###--------------
#
# BUTTON FUNCTIONS
#
###--------------

def selectFile():
  pass

def setChoice():
  pass

# TODO:
# FIGURE OUT WHY YOUR PROGRAM ISN'T RETURNING A TIMELY RESULT
# AND THEN FIGURE OUT WHAT THE HELL CHUNK SEARCHER IS DOING






# Display sample image and annotation segments side by side


# Labeling dialogue

#number_choice = input("CATEGORIES:\n  1. Stomata PRESENT, VISIBLE Pore\n  2. Stomata PRESENT, NO Visible Pore\n  3. Stomata ABSENT\n  4. Incomplete/Ambiguous\n\nCLASSIFY IMAGE: ")

#imwrite(f"sample_images/CAT_{number_choice}-{IMAGE_OBJECT.x1}x-{IMAGE_OBJECT.y1}y-example.tif", chunkToSave)


# Apply random rotations and inversions

# Save subsections to folder with appropriate naming
