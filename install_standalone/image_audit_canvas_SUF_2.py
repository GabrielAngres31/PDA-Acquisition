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
import time
from bresenham import bresenham
# from system_hotkey import SystemHotkey

# TODO: TEMPORARY UNTIL I CAN GET TYPING TO WORK IN PYTHON 3.12
# type EventCoord = dict["x":int, "y":int]
# CURRENTLY UNUSED
class EventCoord :
    # def __init__(self, x:int, y:int, pixel_size:int, margin:int, width=int, height=int):
    #     self.x = x
    #     self.y = y
    #     self.pixel_size = pixel_size
    #     self.margin = margin
    #     self.width = width
    #     self.height = height

    #     self.actual_margin = self.margin        * self.pixel_size
    #     self.actual_width  = self.actual_width  * self.pixel_size
    #     self.actual_height = self.actual_height * self.pixel_size
    #     self.actual_regionsize = self.actual_width + self.actual_margin

    # def disp_x(self):
    #     return self.x // self.pixel_size
    # def disp_y(self):
    #     return self.y // self.pixel_size

    # def x_region(self):
    #     return (self.x - self.actual_margin)//self.actual_regionsize
    # def y_region(self):
    #     return (self.y)//(self.actual_height + self.actual_margin)

    # def arr_x(self):
    #     pass
    # def arr_y(self):
    #     pass

    # def get_disp(self):
    #     pass
    pass



class PixelCanvas:
    def __init__(self, master, width, height, margin=3, base_section_path=None, annot_section_path=None, pixel_size=4):
        
        # Establish main frame
        self.master = master

        # Set initial constants
        self.width = width
        self.height = height
        self.margin = margin

        self.pixel_size = pixel_size

        self.corr_width  = self.width  * self.pixel_size
        self.corr_height = self.height * self.pixel_size
        self.corr_margin = self.margin * self.pixel_size

        self.base_section  = Image.open(base_section_path)
        self.annot_section = Image.open(annot_section_path)
        self.annot_first_mode = self.annot_section.mode
        if self.annot_first_mode == "RGB":
            self.annot_section = self.annot_section.convert("L")        

        self.base_section_array  = np.array(self.base_section)
        # print(self.base_section_array)
        self.annot_section_array = np.array(self.annot_section)
        # print(self.annot_section_array.shape)
        self.drawcolor = 255
        self.overlay_alpha = 40

        # Set up canvases
        self.draw_initial_canvas()
        self.draw_initial_images()


        
        # Create Button-click events
        self.canvas.bind("<Button-1>", self.pixel_multitrack)
        self.canvas.bind("<B1-Motion>", self.pixel_multitrack)
        self.canvas.bind("<ButtonRelease-1>", self.update_overlay)

        self.canvas.bind("<Button-3>", self.pixel_multitrack)
        self.canvas.bind("<B3-Motion>", self.pixel_multitrack)
        self.canvas.bind("<ButtonRelease-3>", self.update_overlay)
        
        self.canvas.bind("<Tab>", lambda event: self.fill_iter(event = event))

        print(self.canvas.bind())

        # Construct base annot canvas       
        for r, row in enumerate(self.annot_section_array):
            for c, col in enumerate(row):
                self.place_pixel(None, c*self.pixel_size + self.corr_margin*2+self.corr_width, r*self.pixel_size+self.corr_margin, self.annot_section_array[r, c]) 


        # Menu Functions for ease of image editing
        self.menubar = tk.Menu(master)
        self.master.config(menu = self.menubar)
        self.mod_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Modify", menu=self.mod_menu)
        self.mod_menu.add_command(label="Clear (WIP)", command=self.clear_annot)
        self.mod_menu.add_command(label="Whiten (WIP)", command=self.set_all_white)

        # hk = SystemHotkey()


        # Last_Pixel
        self.last_pixel_x = 0
        self.last_pixel_y = 0
        

        # Window Exit Handler
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Force Focus
        
        # self.canvas.focus_set()
        # print(repr(self.master.focus_get()))

    

    def set_drawcolor(self, value:int=255):
        assert value < 256 and value >= 0, "Invalid Value for Drawcolor!!"
        self.drawcolor = value

    def draw_initial_canvas(self):
        full_width  = self.corr_width*3 + 4*self.corr_margin
        full_height = self.corr_height  + 2*self.corr_margin
        self.canvas = tk.Canvas(self.master, width=full_width, height=full_height, bg="gray", cursor="plus", borderwidth=1)
        self.canvas.pack()
    
    def draw_initial_images(self):
        base_image = self.base_section.resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST)
        base_image_tk = ImageTk.PhotoImage(base_image)
        self.canvas.create_image(self.corr_margin, self.corr_margin, anchor = tk.NW, image=base_image_tk)
        self.canvas.image_base = base_image_tk
        
        over_image = self.annot_section.resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST)
        over_image_tk = ImageTk.PhotoImage(over_image)
        over_assembly_image = Image.blend(base_image, over_image, self.overlay_alpha/255)
        assemb_img_tk = ImageTk.PhotoImage(over_assembly_image)

        self.canvas.create_image(self.corr_margin*3+self.corr_width*2, self.corr_margin, anchor = tk.NW, image=assemb_img_tk)
        self.canvas.image_over = assemb_img_tk

        self.img_overlay_ann = self.base_section.resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST).convert("RGBA")
        self.img_overlay_bas = self.base_section.resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST).convert("RGBA")

        self.img_overlay_ann.putalpha(self.overlay_alpha)
        self.img_overlay_bas.putalpha(255)

    # Calculates the displayed pixel position of the cursor (self.pixel_size-fold larger than the true pixel size of the monitor).
    def calc_main_coords(self, x, y) -> typing.List:
        main_y = y//self.pixel_size
        main_x = x//self.pixel_size
        # print(f"Calculating from `calc_main_coords({x}, {y}): [x={main_x}, y={main_y}]`")
        return({"main_x":main_x, "main_y":main_y})

    # Returns whether the pixel occupies the first, second, or third canvas
    def calc_region(self, x, y) -> typing.List:
        region_y = y//(self.corr_height + self.corr_margin)
        region_x = (x - self.corr_margin)//(self.corr_width  + self.corr_margin)
        # print(f"Calculating from `calc_region({x}, {y}): [reg_y={region_y}, reg_x={region_x}]`")
        return({"region_x":region_x, "region_y":region_y})
    
    # Calculates whether the pixel is a legitimate drawing point.
    def check_canvas_coords(self, x, y):
        main_x, main_y = x, y
        region = self.calc_region(x, y)
        r_x, r_y = region["region_x"], region["region_y"] 

        if main_x <= self.corr_margin:                     return False
        # if main_x <= self.corr_margin+self.corr_width:     return False
        if main_x >= self.corr_margin*3+self.corr_width*3: return False
        if main_y <= self.corr_margin:                     return False
        if main_y >= self.corr_margin+self.corr_height:    return False

        if main_x >= self.corr_width   + self.corr_margin   and main_x <= self.corr_margin*2 + self.corr_width:   return False
        if main_x >= self.corr_width*2 + self.corr_margin*2 and main_x <= self.corr_margin*3 + self.corr_width*2: return False
        if main_x >= self.corr_width*3 + self.corr_margin*3 and main_x <= self.corr_margin*4 + self.corr_width*3: return False
        
        return True
 
    # Gets array coordinates from canvas px position
    def get_arraycoord_from_canvaspx(self, x, y):
        region = self.calc_region(x, y)
        # print(region)
        main_coords = self.calc_main_coords(x, y)
        # print(main_coords)
        compensate = region["region_x"]*(self.margin+self.width)
        arr_coords = {"arr_x":main_coords["main_x"]-self.margin-compensate, "arr_y":main_coords["main_y"]-self.margin}
        return arr_coords
    
    # Changes an array position at the array coordinates corresponding to the draw coordinates.
    def alter_array(self, x, y, fill=None):
        if fill == None:
            fill = self.drawcolor
        # print(arr_coords)
        arr_coords = self.get_arraycoord_from_canvaspx(x, y)
        self.annot_section_array[arr_coords["arr_y"], arr_coords["arr_x"]] = fill
        # print(self.annot_section_array)

    def place_pixel(self, canvas, x, y, fill=None):
        if fill == None:
            fill_hex = f'{self.drawcolor:02X}'*3
        else:
            fill_hex = f'{fill:02X}'*3
        # print(fill_hex) if fill_hex != "000000" else None
        # print(fill_hex)
        # print(x)
        # print(y)
        self.canvas.create_rectangle(x, y, x+self.pixel_size, y+self.pixel_size, fill="#"+fill_hex, width=0)

    def draw_pixel(self, x, y, draw_fill=None):
        main_coords = self.calc_main_coords(x, y)
        if main_coords["main_x"] == self.last_pixel_x and main_coords["main_y"] == self.last_pixel_y: return
        # print(event.x)
        # print(event.y)
        # print(event.num)
        # print(main_coords)
        if self.check_canvas_coords(x, y):
            self.place_pixel(None, main_coords["main_x"]*self.pixel_size, main_coords["main_y"]*self.pixel_size, draw_fill)
        # print(self.calculate_canvas_coords(self.last_pixel))

    # TODO: CURRENTLY UNUSED AND NONFUNCTIONAL
    # def extrapolate_pixels(self, event):
    #     main_coords = self.calc_main_coords([event.x, event.y])
    #     points = list(bresenham(main_coords["main_x"], main_coords["main_y"], self.last_pixel_y, self.last_pixel_x))
    #     [self.draw_pixel([p[0], p[1]]) for p in points]

    def pixel_multitrack(self, event=None, x=None, y=None):
        # provisional_fill = 96
        # print (event.num)
        if event == None:
            x = x
            y = y
        else:
            x = event.x
            y = event.y
            if event.num == 1:
                self.set_drawcolor(255)
            # provisional_fill = 192
            elif event.num == 3:
                self.set_drawcolor(0)

            # provisional_fill = 32

        ref_coord_base = [x % (self.corr_width+self.corr_margin), y]
        ref_coord_annt = [(x % (self.corr_width+self.corr_margin)) +   self.corr_width +   self.corr_margin, y]
        ref_coord_ovly = [(x % (self.corr_width+self.corr_margin)) + 2*self.corr_width + 2*self.corr_margin, y]
        self.draw_pixel(ref_coord_annt[0], ref_coord_annt[1])
        self.draw_pixel(ref_coord_ovly[0], ref_coord_ovly[1], draw_fill = (192 if self.drawcolor == 255 else 64))
        if self.check_canvas_coords(x, y):
            self.alter_array(x, y)

        
        main_coords = self.calc_main_coords(x, y)

        # self.extrapolate_pixels(event)
        self.last_pixel_x = main_coords["main_x"] 
        self.last_pixel_y = main_coords["main_y"] 
        # time.sleep(0.01)
        return "break"

    def update_overlay(self, event=None):
        base_image = self.base_section.resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST)   
        over_image = Image.fromarray(self.annot_section_array).resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST)
        over_assembly_image = Image.blend(base_image, over_image, self.overlay_alpha/255)
        assemb_img_tk = ImageTk.PhotoImage(over_assembly_image)

        self.canvas.create_image(self.corr_margin*3+self.corr_width*2, self.corr_margin, anchor = tk.NW, image=assemb_img_tk)
        self.canvas.image_over = assemb_img_tk

        self.img_overlay_ann = self.base_section.resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST).convert("RGBA")
        self.img_overlay_bas = self.base_section.resize((self.corr_width, self.corr_height), Image.Resampling.NEAREST).convert("RGBA")

        self.img_overlay_ann.putalpha(self.overlay_alpha)
        self.img_overlay_bas.putalpha(255)

        return "break"

    def clear_annot(self):
        flat_annot = self.annot_section_array.flatten()
        dim = self.annot_section_array.shape[0]
        # corr_factor =  
        [self.place_pixel(None, coord=[p%dim*self.pixel_size+self.corr_margin*2+self.corr_width,p//dim*self.pixel_size+self.corr_margin], fill=0) if flat_annot[p] else None for p in range(len(flat_annot))]
        self.annot_section_array[:,:] = 0
        self.update_overlay()
    
    def set_all_white(self):
        self.annot_section_array[self.annot_section_array != 255] = 0
        # self.annot_section_array[self.annot_section_array != 0] = 255
        # self.annot_section_array[self.annot_section_array > 41] = 255
        self.update_overlay()

    def eff_fill(self, event=None, x=None, y=None, fill=None):
        # Using Flood Fill here because the regions being filled are small
        # print(event)
        # print("yep!")

        pix = self.pixel_size
        if x == None or y == None:
            x, y = event.x, event.y
        arr_coords = self.get_arraycoord_from_canvaspx(x, y)
        # print(arr_coords)
    
        arr_x, arr_y = arr_coords["arr_x"], arr_coords["arr_y"]
        print(f"{x} -> {arr_x}, {y} -> {arr_y}")
        # return
        if fill == None:
            fill = 255-self.annot_section_array[arr_y, arr_x]

        if not self.check_canvas_coords(x, y):
            return

        if self.annot_section_array[arr_y, arr_x] != fill:
            self.annot_section_array[arr_y, arr_x] = fill
            self.place_pixel(None, x*self.pixel_size+self.corr_margin*2, y*self.pixel_size+self.corr_margin, fill=fill)
            if y > 0: self.eff_fill(None, x, y-1)
            if x > 0: self.eff_fill(None, x-1, y)
            if x < self.annot_section_array.shape[1]-1: self.eff_fill(None, x+1, y)
            if y < self.annot_section_array.shape[0]-1: self.eff_fill(None, x, y+1)
    
    def fill_master(self, event, x, y, fill = 0):
        print(f"{x}, {y}")
        if x == None or y == None:
            x, y = event.x, event.y
        if fill == None:
            fill = 255-self.annot_section_array[y, x]

        self.eff_fill(event=event, x=x, y=y)

        self.update_overlay()
        
    def fill_iter(self, event, fill=None):

        # Check if coord is legal to fill from
        can_x, can_y = event.x, event.y
        if not self.check_canvas_coords(can_x, can_y):
            print("You may not fill here!")
            return
        # Establish initial array coordinate
        arr_coords = self.get_arraycoord_from_canvaspx(can_x, can_y)
        arr_x, arr_y = arr_coords["arr_x"], arr_coords["arr_y"]

        # Establish fill color
        if fill == None:
            fill = 255 - self.annot_section_array[arr_y, arr_x]
        
        # Create queue
        points = []
        points.append((arr_x, arr_y))
        # WHILE LOOP BEGINS
        while points:
            x, y = points.pop()
            if (x >= 0 and x < self.width) and (y >= 0 and y < self.height):
                index = [y, x]
                if not self.annot_section_array[index[0], index[1]] == fill:
                    self.annot_section_array[index[0], index[1]] = fill
                    self.draw_pixel(x*self.pixel_size+self.corr_margin*2+self.corr_width, y*self.pixel_size+self.corr_margin, fill)
                    print(arr_x)
                    print(arr_y)
                    # self.draw_pixel(arr_x, arr_y, fill)
                    # print("a")
                    points.append( (x+1,y) )
                    # print("aa")
                    points.append( (x-1,y) )
                    # print("aaa")
                    points.append( (x,y+1) )
                    # print("aaa a")
                    points.append( (x,y-1) )
                    # print("aaaa       a")
                    print(f"plotting px at {x}, {y}")
        self.update_overlay()
        # iterate through queue pop using array coordinates
        # convert popped coord back to canvas for draw_pixel

        # if not self.check_canvas_coords(event.x, event.y): return
        
        # array_coords = self.get_arraycoord_from_canvaspx(event.x, event.y)
        # x, y = array_coords["arr_x"], array_coords["arr_y"]
        
        # fill = 255-self.annot_section_array[y, x]

        # to_update = []

        # to_update.append((x, y))

        # while to_update:
        #     x,y = to_update.pop()   
        #     if (x >= 0 and x < self.width) and (y >= 0 and y < self.height):
        #         index = [y, x]
        #         if not self.annot_section_array[index[0], index[1]] == fill:
        #             self.annot_section_array[index[0], index[1]] = fill
        #             self.pixel_multitrack(None,x*self.pixel_size+self.corr_margin, y*self.pixel_size+self.corr_margin)
        #             print("a")
        #             to_update.append( (x+1,y) )
        #             print("aa")
        #             to_update.append( (x-1,y) )
        #             print("aaa")
        #             to_update.append( (x,y+1) )
        #             print("aaa a")
        #             to_update.append( (x,y-1) )
        #             print("aaaa       a")
        #             print(f"plotting px at {x}, {y}")
        # self.update_overlay()

        pass        
        


    def on_closing(self):
        compare = Image.open("annotation_helper_files/changed_annot_file.png")
        res=messagebox.askyesnocancel('Exit Application', 'Commit changes to image?')
        ## CANCEL
        if res == None:
            messagebox.askokcancel('Return', 'Returning to main application')
        ## YES
        elif res:
            out = Image.fromarray(self.annot_section_array, mode="L")
            out.save("annotation_helper_files/save_annot_file.png")
            if compare == out:
                pass
        ## NO (implicit, doesn't save image)
        self.master.destroy()
    

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_path',
        type = str,
        default = "annotation_helper_files/save_base_file.png",
        help = 'Image section for Base.'
    )
    parser.add_argument(
        '--annot_path',
        type = str,
        default = "annotation_helper_files/save_annot_file.png", # FOR SOME REASON, THIS DOESN'T WORK WITH PNG IMAGES. FIX!!!
        help = 'Image section for Annot.'
    )

    return parser

def main(args:argparse.Namespace) -> bool:
    root = tk.Tk()
    canvas = PixelCanvas(root, 64, 64, base_section_path=args.base_path, annot_section_path=args.annot_path)
    canvas.master.focus_set()
    print(canvas.master.focus_get())
    root.mainloop()

    return True

if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')