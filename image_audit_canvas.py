import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
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
    def __init__(self, master, width, height, margin=3, base_section_path=None, annot_section_path=None, pixel_size=4):
        self.master = master
        self.width = width
        self.height = height
        self.margin = margin
        self.pixel_size = pixel_size
        self.pixel_size_INV = 1/self.pixel_size
        self.corr_height = self.height*self.pixel_size 
        self.corr_width = self.width*self.pixel_size
        self.corr_margin = self.margin*self.pixel_size

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
        


        self.canvas = tk.Canvas(master, width=self.corr_width*3+4*self.corr_margin, height=self.corr_height+2*self.corr_margin, bg="gray", cursor="plus", borderwidth=1)
        self.canvas.pack()
      
        # self.canvas.image = self.testimage

        self.canvas.bind("<Button-1>", self.draw_pixel)
        self.canvas.bind("<B1-Motion>", self.draw_pixel)
        self.canvas.bind("<ButtonRelease-1>", self.draw_pixel)

        self.canvas.bind("<Button-3>", self.draw_pixel)
        self.canvas.bind("<B3-Motion>", self.draw_pixel)
        self.canvas.bind("<ButtonRelease-3>", self.draw_pixel)

        self.base_bound =    {"upper_bound":self.corr_margin, "lower_bound":self.corr_height+self.corr_margin, "left_bound":self.corr_margin,                       "right_bound":self.corr_width  +self.corr_margin}
        self.annot_bound =   {"upper_bound":self.corr_margin, "lower_bound":self.corr_height+self.corr_margin, "left_bound":self.corr_margin*2 + self.corr_width,   "right_bound":self.corr_width*2+self.corr_margin*2}
        self.overlay_bound = {"upper_bound":self.corr_margin, "lower_bound":self.corr_height+self.corr_margin, "left_bound":self.corr_margin*3 + self.corr_width*2, "right_bound":self.corr_width*3+self.corr_margin*3}

        self.base_bound_corr =    dict(zip([key for key in self.base_bound],   [self.base_bound[key]//self.pixel_size    for key in self.base_bound]))
        self.annot_bound_corr =   dict(zip([key for key in self.annot_bound],  [self.annot_bound[key]//self.pixel_size   for key in self.annot_bound]))
        self.overlay_bound_corr = dict(zip([key for key in self.overlay_bound],[self.overlay_bound[key]//self.pixel_size for key in self.overlay_bound]))

        self.draw_initial_canvas()

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def draw_initial_canvas(self):
        # img_base = Image.open(imgpath_dict['base']).resize((self.corr_width, self.corr_height), Image.Resampling.LANCZOS)
        # img_annot = Image.open(imgpath_dict['annot']).resize((self.corr_width, self.corr_height), Image.Resampling.LANCZOS)
        if np.max(self.annot_section_array) == 1:
            array_annot = Image.fromarray(self.annot_section_array*255, mode="L")
        else:
            array_annot = Image.fromarray(self.annot_section_array, mode="RGB") # Weird hacky way to get image to display properly. Image MUST BE JPG

        self.img_base_to_disp =  self.base_section.resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST)
        self.img_annot_to_disp =       array_annot.resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST)

        img_base_tk = ImageTk.PhotoImage(self.img_base_to_disp)
        img_annot_tk = ImageTk.PhotoImage(self.img_annot_to_disp)

        self.img_overlay_ann = self.img_annot_to_disp
        self.img_overlay_bas = self.img_base_to_disp
        self.img_overlay_ann = self.img_overlay_ann.convert("RGBA")
        self.img_overlay_bas = self.img_overlay_bas.convert("RGBA")
        self.img_overlay_ann.putalpha(self.overlay_alpha)
        self.img_overlay_bas.putalpha(255)
        
        self.canvas.create_image(self.corr_margin,                   self.corr_margin, anchor = tk.NW, image=img_base_tk)
        self.canvas.create_image(self.corr_width+self.corr_margin*2, self.corr_margin, anchor = tk.NW, image=img_annot_tk)
        
        self.canvas.image_base=img_base_tk
        self.canvas.image_annot=img_annot_tk

        self.update_overlay(base_img = self.img_overlay_bas, annot_img = self.img_overlay_ann)


    # def update_overlay(self, base_img:Image=None, annot_img:Image=None):
    def update_overlay(self, base_img:Image, annot_img:Image):
        full_overlay = base_img
        # assert base_img.height==height, 3), f"Base Image has improper dimensions: {annot_section_shape} instead of ({width}, {height})"
        full_overlay = Image.blend(full_overlay, annot_img, 0.25)

        full_overlay_tk = ImageTk.PhotoImage(full_overlay)

        self.canvas.create_image(2*self.corr_width+self.corr_margin*3, self.corr_margin, anchor=tk.NW, image=full_overlay_tk)
        self.canvas.image_overlay=full_overlay_tk

    
    def setdrawcolor(self, color):
        self.drawcolor=color
        self.drawcolor_w_alpha = None

    def draw_pixel(self, event):

        if event.num != "??":
            self.pivot_bit = 1-(int(event.num) >> 1)
            self.setdrawcolor(self.color_dict[self.pivot_bit])

        
        # Creates a lattice grid
        x = event.x // self.pixel_size
        y = event.y // self.pixel_size

        if y == self.last_pixel[1]:
            if x == self.last_pixel[0]:
                return None

        if self.corr_margin <= y*self.pixel_size < (self.corr_height+self.corr_margin):

            if self.base_bound_corr['left_bound'] <= x < self.base_bound_corr['right_bound']:
                x1 = (x+self.width) * self.pixel_size + self.corr_margin

            elif self.annot_bound_corr['left_bound'] <= x < self.annot_bound_corr['right_bound']:
                x1 = x * self.pixel_size

            elif self.overlay_bound_corr['left_bound'] <= x < self.overlay_bound_corr['right_bound']:
                x1 = (x-self.width) * self.pixel_size - self.corr_margin

            y1 = y * self.pixel_size
            x2 = x1 + self.pixel_size
            y2 = y1 + self.pixel_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.drawcolor, width=0)
            
            self.last_pixel = [x, y]
            array_x = (x1-(2*self.corr_margin+self.pixel_size*self.width))//self.pixel_size
            array_y = (y1-   self.corr_margin-self.height*self.pixel_size)//self.pixel_size
            
            # print(x1, y1, self.width, self.width*self.pixel_size, array_x, array_y)

            self.annot_section_array_copy_out[array_y][array_x] = [self.pivot_bit*255]*3
            
            annot_copy_update = Image.fromarray(self.annot_section_array_copy_out, mode="RGB")
            annot_copy_update = annot_copy_update.resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST)
            annot_copy_update.convert("RGBA")
            annot_copy_update.putalpha(self.overlay_alpha)
            self.update_overlay(base_img = self.img_overlay_bas, annot_img = annot_copy_update)

    def get_annot_result(self):
        return self.annot__section_array
    
    def on_closing(self):
        res=messagebox.askokcancel('Exit Application', 'Commit changes to image?')
        # print(res)
        if res:
            Image.fromarray(self.annot_section_array_copy_out, mode="RGB").save("annotation_helper_files/changed_annot_file.jpg")
            self.master.destroy()
        else :
            messagebox.askokcancel('Return', 'Returning to main application')
        

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

    # Image.save()

    #return True

if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')

