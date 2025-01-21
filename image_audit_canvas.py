import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import os
# import tkinter_toolkit
# import customtkinter

import typing

import tkinter as tk

class PixelCanvas:
    def __init__(self, master, width, height, margin=12, base_section=None, annot_section=None, pixel_size=4):
        self.master = master
        self.width = width
        self.height = height
        self.margin = margin
        self.pixel_size = pixel_size
        self.pixel_size_INV = 1/self.pixel_size
        self.corr_height = self.height*self.pixel_size 
        self.corr_width = self.width*self.pixel_size

        self.base_section = base_section
        self.annot_section = annot_section

        self.base_section_array = np.array(self.base_section)
        # print(self.base_section)
        self.annot_section_array = np.array(self.annot_section)
        # print(self.annot_section_array)

        # self.color_dict = {1:"white", 0:"black"}
        # self.drawcolor = "white"
        self.color_dict = {1:"#FFFFFF", 0:"#000000"}
        self.drawcolor = "#FFFFFF"
        self.pivotbit = 1
        self.overlay_alpha = 40

        self.last_pixel = [0,0]

        self.matrix_basecanvas = np.zeros((height, width), dtype=int)
        self.matrix_annotcanvas = np.zeros((height, width), dtype=int)
        self.matrix_overlaycanvas = np.zeros((height, width), dtype=int)

        # self.matrix_annotcanvas = np.array()

        # self.testimage = ImageTk.PhotoImage(Image.open("zeta_maxnoise.png"))
        


        self.canvas = tk.Canvas(master, width=self.corr_width*3+4*self.margin, height=self.corr_height+2*self.margin, bg="gray", cursor="plus", borderwidth=1)
        self.canvas.pack()
      
        # self.canvas.image = self.testimage

        self.canvas.bind("<Button-1>", self.draw_pixel)
        self.canvas.bind("<B1-Motion>", self.draw_pixel)
        self.canvas.bind("<ButtonRelease-1>", self.draw_pixel)

        self.canvas.bind("<Button-3>", self.draw_pixel)
        self.canvas.bind("<B3-Motion>", self.draw_pixel)
        self.canvas.bind("<ButtonRelease-3>", self.draw_pixel)

        self.base_bound =    {"upper_bound":self.margin, "lower_bound":self.corr_height+self.margin, "left_bound":self.margin,                       "right_bound":self.corr_width  +self.margin}
        self.annot_bound =   {"upper_bound":self.margin, "lower_bound":self.corr_height+self.margin, "left_bound":self.margin*2 + self.corr_width,   "right_bound":self.corr_width*2+self.margin*2}
        self.overlay_bound = {"upper_bound":self.margin, "lower_bound":self.corr_height+self.margin, "left_bound":self.margin*3 + self.corr_width*2, "right_bound":self.corr_width*3+self.margin*3}

        self.base_bound_corr =    dict(zip([key for key in self.base_bound],   [self.base_bound[key]//self.pixel_size    for key in self.base_bound]))
        self.annot_bound_corr =   dict(zip([key for key in self.annot_bound],  [self.annot_bound[key]//self.pixel_size   for key in self.annot_bound]))
        self.overlay_bound_corr = dict(zip([key for key in self.overlay_bound],[self.overlay_bound[key]//self.pixel_size for key in self.overlay_bound]))

        self.draw_initial_canvas()

    # def placeholder_drawimage(self, imgpath):
    #     if os.path.exists(imgpath):
    #         img = ImageTk.PhotoImage(Image.open(imgpath).resize((self.corr_width*2,self.corr_height*2), Image.Resampling.LANCZOS))

    #         img_placeholder_BASE =    self.canvas.create_image(  self.corr_width +  self.margin, self.margin, anchor=tk.NW, image=img)
    #         img_placeholder_ANNOT =   self.canvas.create_image(2*self.corr_width +3*self.margin, self.margin, anchor=tk.NW, image=img)
    #         img_placeholder_OVERLAY = self.canvas.create_image(3*self.corr_width +5*self.margin, self.margin, anchor=tk.NW, image=img)
    #         # TODO: How do i...change the image position




    #         self.canvas.image = img
    #     else:
    #         print(f"Error: Image file not found: {imgpath}")

    
    # def draw_initial_canvas(self, imgpath_dict):
    def draw_initial_canvas(self):
        # img_base = Image.open(imgpath_dict['base']).resize((self.corr_width, self.corr_height), Image.Resampling.LANCZOS)
        # img_annot = Image.open(imgpath_dict['annot']).resize((self.corr_width, self.corr_height), Image.Resampling.LANCZOS)
        img_base =  self.base_section.resize(( self.corr_width, self.corr_height), Image.Resampling.LANCZOS)
        img_annot = self.annot_section.resize((self.corr_width, self.corr_height), Image.Resampling.LANCZOS)
        img_base_tk = ImageTk.PhotoImage(img_base)
        img_annot_tk = ImageTk.PhotoImage(img_annot)
        self.canvas.create_image(self.margin,                   self.margin, anchor = tk.NW, image=img_base_tk)
        self.canvas.create_image(self.corr_width+self.margin*2, self.margin, anchor = tk.NW, image=img_annot_tk)


        img_overlay_ann = img_annot
        img_overlay_bas = img_base
        img_overlay_ann = img_overlay_ann.convert("RGBA")
        img_overlay_bas = img_overlay_bas.convert("RGBA")
        img_overlay_ann.putalpha(self.overlay_alpha)
        img_overlay_bas.putalpha(255)


        full_overlay = img_overlay_bas

        full_overlay = Image.blend(full_overlay, img_overlay_ann, 0.25)

        full_overlay_tk = ImageTk.PhotoImage(full_overlay)

        self.canvas.create_image(2*self.corr_width+self.margin*3, self.margin, anchor=tk.NW, image=full_overlay_tk)

        self.canvas.image_base=img_base_tk
        self.canvas.image_annot=img_annot_tk
        self.canvas.image_overlay=full_overlay_tk

    
    def setdrawcolor(self, color):
        self.drawcolor=color
        self.drawcolor_w_alpha = None

    def draw_pixel(self, event):

        if event.num != "??":
            self.pivot_bit = 1-(int(event.num) >> 1)
            self.setdrawcolor(self.color_dict[self.pivot_bit])

        x = event.x // self.pixel_size
        y = event.y // self.pixel_size

        if y == self.last_pixel[1]:
            if x == self.last_pixel[0]:
                return None

        if self.margin <= y*self.pixel_size < (self.corr_height+self.margin):

            if self.base_bound_corr['left_bound'] <= x < self.base_bound_corr['right_bound']:
                x1 = (x+self.width) * self.pixel_size + self.margin
                y1 = y * self.pixel_size
                x2 = x1 + self.pixel_size
                y2 = y1 + self.pixel_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.drawcolor, width=0)

            elif self.annot_bound_corr['left_bound'] <= x < self.annot_bound_corr['right_bound']:
                x1 = x * self.pixel_size
                y1 = y * self.pixel_size
                x2 = x1 + self.pixel_size 
                y2 = y1 + self.pixel_size 
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.drawcolor, width=0)

            elif self.overlay_bound_corr['left_bound'] <= x < self.overlay_bound_corr['right_bound']:
                x1 = (x-self.width) * self.pixel_size - self.margin
                y1 = y * self.pixel_size
                x2 = x1 + self.pixel_size
                y2 = y1 + self.pixel_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.drawcolor, width=0)

            

            self.last_pixel = [x, y]
            self.annot_section_array[int(y-self.margin/self.pixel_size)%self.height][int(x-self.margin/self.pixel_size)%self.width] = self.pivot_bit

root = tk.Tk()
canvas = PixelCanvas(root, 64, 64, base_section=Image.open('test_stomata_viz_BASE.png'), annot_section=Image.open('test_stomata_viz_ANNOT.png'))
np.set_printoptions(threshold=np.inf)
root.mainloop()