import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from PIL import Image, ImageTk
import os
import image_audit_canvas_SUF_2
import subprocess
import clean_image_SUF as cl
import clumps_table_SUF as cf
import numpy as np


def x0(bbox_coords): 
    """
    Returns the LEFT boundary of a given clump's bounding box.
    """
    return bbox_coords[0]

def y0(bbox_coords):
    """
    Returns the UPPER boundary of a given clump's bounding box.
    """
    return bbox_coords[1]

def x1(bbox_coords): 
    """
    Returns the RIGHT boundary of a given clump's bounding box.
    """
    return bbox_coords[2]

def y1(self):
    """
    Returns the BOTTOM boundary of a given clump's bounding box.
    """
    return -self.bbox_coords[3]

def xc(self):
    """
    Returns the X-Midpoint (horizontal) of a given clump's bounding box.
    """
    return (self.x0()+self.x1())//2

def yc(self):
    """
    Returns the Y-Midpoint (vertical) of a given clump's bounding box.
    """
    return (self.y0()+self.y1())//2

# Fixed-size window centered on chunk, upper left corner
def xD(self):
    """
    X-coordinate of upper-left corner of NxN frame centered on chunk.
    """
    return self.xc()+self.window_sidelength//2

def yD(self):
    """
    Y-coordinate of upper-left corner of NxN frame centered on chunk.
    """
    return self.yc()+self.window_sidelength//2