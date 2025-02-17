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
        print(self.annot_section_array_copy_out.shape)

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

        self.menubar = tk.Menu(master)
        self.master.config(menu = self.menubar)
        self.mod_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Modify", menu=self.mod_menu)
        self.mod_menu.add_command(label="Clear (WIP)", command=self.clear_annot)
        self.mod_menu.add_command(label="Whiten (WIP)", command=self.set_all_white)

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
            # annot_section_threshold = np.where(self.annot_section_array*255 > 27, 255, 0)
            annot_section_threshold = np.where(self.annot_section_array*255 > 42, self.annot_section_array, 0)
            # array_annot = Image.fromarray(self.annot_section_array, mode="L")
            array_annot = Image.fromarray(annot_section_threshold, mode="L")
            print("A")
        else:
            if self.annot_section_array.shape[1] != 3:
                annot_section_threshold = np.where(self.annot_section_array > 42, self.annot_section_array, 0)
                annot_section_threshold = annot_section_threshold[:,:,0]
                print(annot_section_threshold.shape)

        # print(self.annot_section_array.shape)
        print("groat")
        print(annot_section_threshold.shape)
        # annot_section_threshold = np.where(self.annot_section_array > 27, self.annot_section_array, 0)
        # array_annot = Image.fromarray(self.annot_section_array, mode="RGB") # Weird hacky way to get image to display properly. Image MUST BE JPG
        array_annot = Image.fromarray(annot_section_threshold, mode="L") # Weird hacky way to get image to display properly. Image MUST BE JPG
        print("B")
    
        

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
        self.set_all_white()


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
            # print(self.annot_section_array_copy_out.shape)
            
            annot_copy_update = Image.fromarray(self.annot_section_array_copy_out, mode="RGB")
            annot_copy_update = annot_copy_update.resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST)
            annot_copy_update.convert("RGBA")
            annot_copy_update.putalpha(self.overlay_alpha)
            self.update_overlay(base_img = self.img_overlay_bas, annot_img = annot_copy_update)

    def get_annot_result(self):
        return self.annot__section_array
    
    def on_closing(self):
        res=messagebox.askyesnocancel('Exit Application', 'Commit changes to image?')
        
        ## CANCEL
        if res == None:
            messagebox.askokcancel('Return', 'Returning to main application')
        ## YES
        elif res:
            Image.fromarray(self.annot_section_array_copy_out, mode="RGB").save("annotation_helper_files/changed_annot_file.jpg")
        
        ## NO or YES
        self.master.destroy()
    
    def clear_annot(self):
        self.annot_section_array_copy_out = np.zeros_like(self.annot_section_array_copy_out)
        annot_clear_update = Image.fromarray(self.annot_section_array_copy_out, mode="RGB")
        annot_clear_update = annot_clear_update.resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST)
        annot_clear_update.convert("RGBA")
        annot_clear_update.putalpha(self.overlay_alpha)
        self.update_overlay(base_img = self.img_overlay_bas, annot_img = annot_clear_update)
        
        # [print(x, y, x+self.pixel_size, y+self.pixel_size) for x, y in [(x,y) for x in range(self.annot_bound_corr['left_bound'], self.annot_bound_corr['right_bound'], self.pixel_size) for y in range(self.corr_margin, self.corr_height+self.corr_margin, self.pixel_size)]]
        [self.canvas.create_rectangle(x, y, x+self.pixel_size, y+self.pixel_size, fill="black", width=0) for x, y in [(x,y) for x in range(self.annot_bound_corr['left_bound']*self.pixel_size, self.annot_bound_corr['right_bound']*self.pixel_size, self.pixel_size) for y in range(self.corr_margin, self.corr_height+self.corr_margin, self.pixel_size)]]

    def set_all_white(self):
        self.annot_section_enwhiten = self.annot_section_array_copy_out
        self.annot_section_enwhiten[self.annot_section_enwhiten < 80] = 0
        
        h, w = self.annot_section_array_copy_out.shape[0], self.annot_section_array_copy_out.shape[1]

        # print(h)
        # print(w)
        annot_white_update = Image.fromarray(self.annot_section_enwhiten, mode="RGB")
        annot_white_update = annot_white_update.resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST)
        annot_white_update.convert("RGBA")
        annot_white_update.putalpha(self.overlay_alpha)
        self.update_overlay(base_img = self.img_overlay_bas, annot_img = annot_white_update)

        # [self.canvas.create_rectangle(x, y, x+self.pixel_size, y+self.pixel_size, fill="black", width=0) for x, y in [(x,y) for x in range(self.annot_bound_corr['left_bound']*self.pixel_size, self.annot_bound_corr['right_bound']*self.pixel_size, self.pixel_size) for y in range(self.corr_margin, self.corr_height+self.corr_margin, self.pixel_size)]]
        for coords in [(x,y) for x in range(w) for y in range(h)]:
            # print(f'{coords}: x={coords[0]}, y={coords[1]}')
            # print(coords)
            
            x = coords[0]
            y = coords[1]
            # print(self.annot_section_enwhiten[y][x][:])
            # print(self.annot_section_enwhiten[y][x][0])
            x_canvas = (self.annot_bound_corr['left_bound']+x)*self.pixel_size
            y_canvas = self.corr_margin+y*self.pixel_size
            fill = ("black" if (not self.annot_section_enwhiten[y][x][0]) else "white")
            # fill = ("white" if (not self.annot_section_enwhiten[y][x][0]) else "black")
            self.canvas.create_rectangle(x_canvas, y_canvas, x_canvas+self.pixel_size, y_canvas+self.pixel_size, fill=fill, width=0) 
            try:
                assert (fill == "white" and self.annot_section_enwhiten[y][x][0] > 0) or (fill == "black" and self.annot_section_enwhiten[y][x][0] == 0), "lol lmoaf hahahahahaha"
                assert x < 64
                assert y < 64
            except AssertionError:
                print("hell")
        # [self.canvas.create_rectangle(x, y, x+self.pixel_size, y+self.pixel_size, fill="black", width=0) for x, y in [(x,y) for x in range(self.annot_bound_corr['left_bound']*self.pixel_size, self.annot_bound_corr['right_bound']*self.pixel_size, self.pixel_size) for y in range(self.corr_margin, self.corr_height+self.corr_margin, self.pixel_size)]]
        

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

