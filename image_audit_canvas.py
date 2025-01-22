import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import os
# import tkinter_toolkit
# import customtkinter

import typing

import tkinter as tk
import argparse

np.set_printoptions(threshold=np.inf)

class PixelCanvas:
    def __init__(self, master, width, height, margin=12, base_section_path=None, annot_section_path=None, pixel_size=4):
        self.master = master
        self.width = width
        self.height = height
        self.margin = margin
        self.pixel_size = pixel_size
        self.pixel_size_INV = 1/self.pixel_size
        self.corr_height = self.height*self.pixel_size 
        self.corr_width = self.width*self.pixel_size

        self.base_section = Image.open(base_section_path)
        self.annot_section = Image.open(annot_section_path)

        self.base_section_array = np.array(self.base_section)
        # print(self.base_section)
        self.annot_section_array = np.asarray(self.annot_section, dtype=np.uint8)
        self.annot_section_array_copy_out = np.copy(self.annot_section_array)

        base_section_shape = self.base_section_array.shape
        annot_section_shape = self.annot_section_array.shape
        assert base_section_shape==(width, height), f"Base Image has improper dimensions: {base_section_shape} instead of ({width}, {height})"
        assert annot_section_shape==(width, height, 3), f"Base Image has improper dimensions: {annot_section_shape} instead of ({width}, {height})"
        print("annot_section_array")
        print(self.annot_section_array.shape)
        # self.annot_section.show()
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

    def draw_initial_canvas(self):
        # img_base = Image.open(imgpath_dict['base']).resize((self.corr_width, self.corr_height), Image.Resampling.LANCZOS)
        # img_annot = Image.open(imgpath_dict['annot']).resize((self.corr_width, self.corr_height), Image.Resampling.LANCZOS)
        if np.max(self.annot_section_array) == 1:
            array_annot = Image.fromarray(self.annot_section_array*255, mode="L")
        else:
            array_annot = Image.fromarray(self.annot_section_array, mode="RGB") # Weird hacky way to get image to display properly. Image MUST BE JPG


        # array_annot.show()

        img_base =  self.base_section.resize(( self.corr_width, self.corr_height), Image.Resampling.NEAREST)
        # img_annot = self.annot_section.resize((self.corr_width, self.corr_height), Image.Resampling.LANCZOS)
        img_annot = array_annot.resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST)

        self.img_base_tk = ImageTk.PhotoImage(img_base)
        self.img_annot_tk = ImageTk.PhotoImage(img_annot)
        self.canvas.create_image(self.margin,                   self.margin, anchor = tk.NW, image=self.img_base_tk)
        self.canvas.create_image(self.corr_width+self.margin*2, self.margin, anchor = tk.NW, image=self.img_annot_tk)


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

        self.canvas.image_base=self.img_base_tk
        self.canvas.image_annot=self.img_annot_tk
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
            # print(self.annot_section_array_copy_out)
            self.annot_section_array_copy_out[int(y-self.margin/self.pixel_size)%self.height][int(x-self.margin/self.pixel_size)%self.width] = [self.pivot_bit]*3

            img_overlay_ann = self.img_annot
            # img_overlay_bas = self.img_base
            img_overlay_ann = img_overlay_ann.convert("RGBA")
            # img_overlay_bas = img_overlay_bas.convert("RGBA")
            img_overlay_ann.putalpha(self.overlay_alpha)
            img_overlay_bas.putalpha(255)


            full_overlay = img_overlay_bas

            full_overlay = Image.blend(full_overlay, img_overlay_ann, 0.25)

            full_overlay_tk = ImageTk.PhotoImage(full_overlay)

            self.canvas.create_image(2*self.corr_width+self.margin*3, self.margin, anchor=tk.NW, image=full_overlay_tk)

        

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_path',
        type = str,
        default = "test_stomata_viz_BASE.png",
        #required = True,
        help = 'Image section for Base.'
    )
    parser.add_argument(
        '--annot_path',
        type = str,
        default = "fuzzy_test.jpg",
        #required = True,
        help = 'Image section for Annot.'
    )

    return parser

def main(args:argparse.Namespace) -> bool:
    
    root = tk.Tk()
    # canvas = PixelCanvas(root, 64, 64, base_section=Image.open('test_stomata_viz_BASE.png'), annot_section=Image.open('test_stomata_viz_ANNOT.jpg'))
    # canvas = PixelCanvas(root, 64, 64, base_section=Image.open('test_stomata_viz_BASE.png'), annot_section=Image.open('fuzzy_test.jpg'))
    canvas = PixelCanvas(root, 64, 64, base_section_path=args.base_path, annot_section_path=args.annot_path)
    root.mainloop()

    Image.save()

    #return True

if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')

