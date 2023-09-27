from random import randint

import matplotlib.pyplot as plt
from PIL import Image

from tifffile import imread, imwrite

import os
CWD = os.getcwd()
IMAG_DIR = os.path.join(CWD, "SCD_training_data")
TEMP_DIR = os.path.join(CWD, "SCD_training_data\\temp_files")
print("DIR: " + TEMP_DIR)
import numpy as np

from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk

import copy

import time


class MANAGER_images:
  def __init__(self, segment_size, buffer_size, 
               filename_BASE_image = os.path.join(IMAG_DIR, "cot1.tif"),
               filename_ANNO_image = os.path.join(IMAG_DIR, "cot1_MASK.tif")):

      self.filename_BASE_image = filename_BASE_image
      self.filename_ANNO_image = filename_ANNO_image

      self.image_BASE = np.asarray(Image.open(self.filename_BASE_image))
      self.image_ANNO = np.asarray(Image.open(self.filename_ANNO_image))

      self.photoimageBASE = ImageTk.PhotoImage(Image.fromarray(self.image_BASE))
      self.photoimageANNO = ImageTk.PhotoImage(Image.fromarray(self.image_ANNO))

      self.segment_size = segment_size
      self.buffer_size = buffer_size

      self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0
      self.cornerInit = [0, 0]
      self.cornerFinal= [0, 0]  


   #def setSourceImage(self, choice):
    
   # val_file_choice = "filename_" + choice + "_image"
   # setattr(self, val_file_choice, fd.askopenfilename())
   # print(f"{getattr(self, val_file_choice)} file set")

   # val_file_actual = imread( getattr(self, "filename_" + choice + "_image" ) )
   # setattr(self, "image_" + choice, val_file_actual)
   # print(f"{choice} image loaded")
   # print(type(val_file_actual))
  
  def setGivenArea(self):
    assert self.image_BASE.shape[0] == self.image_ANNO.shape[0]
    assert self.image_BASE.shape[1] == self.image_ANNO.shape[1]

    self.x1 = randint(0, self.image_BASE.shape[0] - self.segment_size - 1)
    self.y1 = randint(0, self.image_BASE.shape[1] - self.segment_size - 1)

    self.x2 = self.x1 + self.segment_size - 1
    self.y2 = self.y1 + self.segment_size - 1

    self.cornerInit  = [self.x1, self.y1]
    self.cornerFinal = [self.x2, self.y2]

    print(self.cornerInit)
    print(self.cornerFinal)

def drawBox(canvas_obj, size, buffer, color="blue"):

    canvas_obj.create_line(buffer,      buffer,      buffer+size, buffer,      fill = color)
    canvas_obj.create_line(buffer,      buffer,      buffer,      buffer+size, fill = color)
    canvas_obj.create_line(buffer+size, buffer,      buffer+size, buffer+size, fill = color)
    canvas_obj.create_line(buffer,      buffer+size, buffer+size, buffer+size, fill = color)




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

window.resizable(width=False, height=False)

IMAGE_MANAGER = MANAGER_images(64, 20)

# FRAME CONSTRUCTION

# Make the window frame
coreframe = Frame(window)
coreframe.pack()

# Make the housing for the two image snippets
FRAME_imageComparison = Frame(coreframe, bg = "grey")
FRAME_imageComparison.pack(side = TOP, fill = X, expand = False)

# Make the housing for the load file buttons
FRAME_actionButtons = Frame(coreframe, bg = "yellow")
FRAME_actionButtons.pack(side = BOTTOM, fill = X, expand = False, pady = 12)

# Split the frame area for the two images
SUBFRAME_baseImage = Frame(FRAME_imageComparison, bg = "blue")
SUBFRAME_baseImage.pack(side = LEFT, fill = X, expand = False)

# Draw the BASE Canvas
CANVAS_BASE = Canvas(SUBFRAME_baseImage, width = 104, height = 104, background = "black")
CANVAS_BASE.pack(side=TOP)

# THIS RAN!!!
imgpathBASE = os.path.join(IMAG_DIR, "cot1.tif")
imgNPBASE = np.asarray(Image.open(imgpathBASE))
imgphotoBASE = ImageTk.PhotoImage(image=Image.fromarray(imgNPBASE))

placeholder_image_base = imgphotoBASE
#placeholder_image_base = ImageTk.PhotoImage(Image.open(os.path.join(TEMP_DIR, "IMG_1058.jpg")))
#placeholder_image_test = ImageTk.PhotoImage(Image.open(os.path.join(TEMP_DIR, "8501_GMI.jpg")))
CANVAS_BASE.create_image(-1500, -1500, anchor=NW, image=placeholder_image_base)

# Draw the ANNO Canvas
SUBFRAME_annoImage = Frame(FRAME_imageComparison, bg = "orange")
SUBFRAME_annoImage.pack(side = RIGHT, fill = X, expand = False)

CANVAS_ANNO = Canvas(SUBFRAME_annoImage, width = 104, height = 104, background = "black")
CANVAS_ANNO.pack(side=TOP)

placeholder_image_anno = ImageTk.PhotoImage(Image.open(os.path.join(TEMP_DIR, "8501_GMI.jpg")))
CANVAS_ANNO.create_image(20, 20, anchor=NW, image=placeholder_image_anno)

# Buttons that open file browser and allow you to select two images to use for selection

#BUTTON_selectBase = Button(SUBFRAME_baseImage, text = "Open Base Image", fg = "black", command = lambda: IMAGE_MANAGER.setSourceImage("BASE"))
#BUTTON_selectBase.pack(side=BOTTOM, fill = X)

#BUTTON_selectAnno = Button(SUBFRAME_annoImage, text = "Open Annotation", fg = "black", command = lambda: IMAGE_MANAGER.setSourceImage("ANNO"))
#BUTTON_selectAnno.pack(side=BOTTOM, fill = X)

# Make the housing for the Generator Buttons
SUBFRAME_generatorButtons = Frame(FRAME_actionButtons)
SUBFRAME_generatorButtons.pack(side = LEFT, fill = BOTH, expand = False)

# Make the housing for the Decision Buttons
SUBFRAME_decisionsButtons = Frame(FRAME_actionButtons)
SUBFRAME_decisionsButtons.pack(side = RIGHT, fill = BOTH, expand = False)

# Make Image generation button/s
print(CWD)
base_img = Image.open(os.path.join(TEMP_DIR, "IMG_1058.jpg"))
anno_img = Image.open(os.path.join(TEMP_DIR, "8501_GMI.jpg"))
# Default values ^^ are a picture of my cat

# Set template images
base_box = ImageTk.PhotoImage(base_img)
anno_box = ImageTk.PhotoImage(anno_img)

def reloadCanvas(mgr, cvs, img):
  mgr.setGivenArea()
  cvs.delete('all')
  #imgpathBASE = os.path.join(IMAG_DIR, "cot1.tif")
  #imgNPBASE = np.asarray(Image.open(imgpathBASE))
  #imgphotoBASE = ImageTk.PhotoImage(image=Image.fromarray(imgNPBASE))

  imgpath = os.path.join(IMAG_DIR, img)
  imgNP = np.asarray(Image.open(imgpath))
  imgphoto = ImageTk.PhotoImage(image=Image.fromarray(imgNP))
  cvs.create_image(-2000, -2000, anchor=NW, image = imgphoto)
  drawBox(cvs, 64, 20)


def reloadPreviews():
  global IMAGE_MANAGER
  #global label_base
  #global label_anno
  global window
  global CANVAS_BASE
  global CANVAS_ANNO
  #CANVAS_BASE.delete()
  #CANVAS_BASE.update_idletasks()
  #CANVAS_ANNO.delete()
  #CANVAS_ANNO.update_idletasks()

  CANVAS_BASE.delete('all')
  #imageBASE = Image.open(IMAGE_MANAGER.filename_BASE_image)
  #imgBASE = ImageTk.PhotoImage(image = imageBASE)

  imgpathBASE = os.path.join(IMAG_DIR, "cot1.tif")
  imgNPBASE = np.asarray(Image.open(imgpathBASE))
  imgphotoBASE = ImageTk.PhotoImage(image=Image.fromarray(imgNPBASE))
  
  
  print("ATTEMPTING DISPLAY")
  base_inst = CANVAS_BASE.create_image(1500, 1500, anchor=NW, 
                                       image=imgphotoBASE)

  print(imgphotoBASE)

  print("FLAG")
  #print(ImageTk.PhotoImage(Image.fromarray(IMAGE_MANAGER.image_BASE)))

  print(str(IMAGE_MANAGER.x1) + " X - " + str(IMAGE_MANAGER.y1) + " Y")
  drawBox(CANVAS_BASE, 64, 20)

  print("Created BASE")
  CANVAS_BASE.update()

  CANVAS_ANNO.delete('all')

  anno_img_holder = image=ImageTk.PhotoImage(file = IMAGE_MANAGER.filename_ANNO_image)
  anno_inst = CANVAS_ANNO.create_image(1500, 1500, anchor=NW, 
                                       #image=ImageTk.PhotoImage(Image.fromarray(image_manager.image_ANNO))
                                       #image = ImageTk.PhotoImage(Image.open(os.path.join(TEMP_DIR, "test_tif.tif")))
                                       image = anno_img_holder
                                       )  


  #print(image_manager.image_ANNO)
  #print(Image.fromarray(image_manager.image_ANNO))
  #print(ImageTk.PhotoImage(Image.fromarray(image_manager.image_ANNO)))
  print(anno_inst)
  drawBox(CANVAS_ANNO, 64, 20)

  CANVAS_ANNO.update()

  print("Created ANNO")
  
  #CANVAS_ANNO.update_idletasks()
  #label_anno.config(image=anno_box)
  #CANVAS_BASE.update_idletasks()


### Unbiased Random
BUTTON_unbiasedRandom = Button(SUBFRAME_generatorButtons, text = "Random Image", fg = "black",
                               command = lambda: reloadCanvas(IMAGE_MANAGER, CANVAS_BASE, "cot1.tif"))
                                  #CANVAS_BASE.delete('all'),
                                  #CANVAS_BASE.create_image(-15, -15, anchor=NW, 
                                  #     image=ImageTk.PhotoImage(image = ImageTk.PhotoImage(image=Image.fromarray(np.asarray(Image.open("test_tif.tif")))))),
                                  #drawBox(CANVAS_BASE, 64, 20)
                                  #)

                               #command = lambda: [IMAGE_OBJECT.chunkSearcher("Unbiased Random"),
                               #                   IMAGE_OBJECT.imagePrepare(),
                               #                   reloadPreviews(IMAGE_OBJECT)])
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
