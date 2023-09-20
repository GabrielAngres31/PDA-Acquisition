#import keras
#import tensorflow as tf

import random

import matplotlib.pyplot as plt

import os

import numpy as np

def monoToRGB(img):
  return np.array([[[i, i, i] for i in j] for j in img])

class SAMPLER_IMAGE:
  def __init__(self, BASE, MASK, segment_size, buffer_size):

    self.BASE = np.array(monoToRGB(BASE))
    self.MASK = np.array(MASK)

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
    self.x1 = random.randint(0, self.size[0] - self.segment_size - 1)
    self.y1 = random.randint(0, self.size[1] - self.segment_size - 1)

    self.x2 = self.x1 + self.segment_size - 1
    self.y2 = self.y1 + self.segment_size - 1

    self.cornerInit = [self.x1, self.y1]
    self.cornerFinal = [self.x2, self.y2]

  def grabChunk(self, choice):
    if choice == "BASE":
      return self.BASE[(self.y1-self.buffer_size):(self.y2+self.buffer_size), (self.x1-self.buffer_size):(self.x2+self.buffer_size)]
    if choice == "MASK":
      return self.MASK[(self.y1-self.buffer_size):(self.y2+self.buffer_size), (self.x1-self.buffer_size):(self.x2+self.buffer_size)]

IMAGE_OBJECT = SAMPLER_IMAGE(BASE = plt.imread("C:/Users/gjang/Documents/Muroyama Lab Studies/Cotyledon Imaging Tiffs/SCD training data/120621_cot1_max_rotated_c2.tif"),
                             MASK = plt.imread("C:/Users/gjang/Documents/Muroyama Lab Studies/Cotyledon Imaging Tiffs/SCD training data/120621_cot1_max_rotated_c2_MASK.tif"),
                             segment_size = 64,
                             buffer_size = 20)

def drawBox(IMAGE, corner_one, corner_two, color=(0, 0, 255)):
  x1, y1 = corner_one
  x2, y2 = corner_two
  # Draw top and bottom edges
  for x in range(x1, x2+1):
    IMAGE[y1][x] = color
    IMAGE[y2][x] = color
  # Draw left and right edges
  for y in range(y1+1, y2):
    IMAGE[y][x1] = color
    IMAGE[y][x2] = color

IMAGE_OBJECT.setGivenArea()

drawBox(IMAGE_OBJECT.BASE, IMAGE_OBJECT.cornerInit, IMAGE_OBJECT.cornerFinal)

### figure_mask = plt.figure("MASK")

drawBox(IMAGE_OBJECT.MASK, IMAGE_OBJECT.cornerInit, IMAGE_OBJECT.cornerFinal)

# Display sample image and annotation segments side by side
plt.figure(1)
plt.subplot(121)
plt.imshow(IMAGE_OBJECT.grabChunk("BASE"), origin = "lower")
plt.subplot(122)
plt.imshow(IMAGE_OBJECT.grabChunk("MASK"), origin = "lower")
plt.show()

# Labeling dialogue

# Apply random rotations and inversions

# Save subsections to folder with appropriate naming
