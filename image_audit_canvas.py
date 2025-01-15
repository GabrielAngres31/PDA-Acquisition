import matplotlib.pyplot as plt
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import os
# import tkinter_toolkit
# import customtkinter

import typing

class audit_canvas():
    pass


import tkinter as tk

class PixelCanvas:
    def __init__(self, master, width, height, margin=16, base_section=None, annot_section=None, pixel_size=4):
        self.master = master
        self.width = width
        self.height = height
        self.margin = margin
        self.pixel_size = pixel_size
        self.corr_height = self.height*self.pixel_size 
        self.corr_width = self.width*self.pixel_size

        self.matrix_basecanvas = np.zeros((height, width), dtype=int)
        self.matrix_annotcanvas = np.zeros((height, width), dtype=int)
        self.matrix_overlaycanvas = np.zeros((height, width), dtype=int)

        # self.testimage = ImageTk.PhotoImage(Image.open("zeta_maxnoise.png"))
        
        self.drawcolor = "white"
        self.pivotbit = 1

        self.canvas = tk.Canvas(master, width=self.corr_width*3.25, height=self.corr_height+2*self.margin, bg="gray", cursor="plus", borderwidth=1)
        self.canvas.pack()
      
        # self.canvas.image = self.testimage

        self.canvas.bind("<Button-1>", self.draw_pixel)
        self.canvas.bind("<B1-Motion>", self.draw_pixel)
        self.canvas.bind("<ButtonRelease-1>", self.draw_pixel)

        self.canvas.bind("<Button-3>", self.draw_pixel)
        self.canvas.bind("<B3-Motion>", self.draw_pixel)
        self.canvas.bind("<ButtonRelease-3>", self.draw_pixel)

    def placeholder_drawimage(self, imgpath):
        if os.path.exists(imgpath):
            img = ImageTk.PhotoImage(Image.open(imgpath).resize((self.corr_width*2,self.corr_height*2), Image.Resampling.LANCZOS))

            img_placeholder_BASE = self.canvas.create_image(self.corr_width+self.margin, self.margin, anchor=tk.NW, image=img)
            img_placeholder_ANNOT = self.canvas.create_image(2*self.corr_width+3*self.margin, self.margin, anchor=tk.NW, image=img)
            img_placeholder_OVERLAY = self.canvas.create_image(3*self.corr_width+5*self.margin, self.margin, anchor=tk.NW, image=img)
            # TODO: How do i...change the image position




            self.canvas.image = img
        else:
            print(f"Error: Image file not found: {imgpath}")

    
    def draw_initial_canvas(self, imgpath_dict):
        # def prep_image(img_path):
        #     img = ImageTk.PhotoImage(Image.open(img_path).resize((self.corr_width*2,self.corr_height*2), Image.Resampling.LANCZOS))
        #     self.canvas.image = img
        #     return img
        
        # img_placeholder_BASE = self.canvas.create_image(self.corr_width+10, 0, image=prep_image(imgpath_dict['base']))
        # img_placeholder_ANNOT = self.canvas.create_image(2*self.corr_width+20, 0, image=prep_image(imgpath_dict['annot']))

        # self.canvas.image = img_placeholder_BASE
        # img_placeholder_OVERLAY = self.canvas.create_image(3*self.corr_width+30, 0, image=prep_image(imgpath_dict['overlay']))
        img_base = Image.open(imgpath_dict['base']).resize((self.corr_width, self.corr_height), Image.Resampling.LANCZOS)
        img_annot = Image.open(imgpath_dict['annot']).resize((self.corr_width, self.corr_height), Image.Resampling.LANCZOS)
        img_base_tk = ImageTk.PhotoImage(img_base)
        img_annot_tk = ImageTk.PhotoImage(img_annot)
        self.canvas.create_image(self.margin, self.margin, anchor = tk.NW, image=img_base_tk)
        self.canvas.create_image(self.corr_width+self.margin*2, self.margin, anchor = tk.NW, image=img_annot_tk)
        # self.canvas.create_image(2*self.corr_width+20, 0, anchor = tk.NW, image=img_base_tk)

        # overlay_func = lambda b, a: 

        img_overlay_ann = img_annot
        img_overlay_bas = img_base
        img_overlay_ann = img_overlay_ann.convert("RGBA")
        img_overlay_bas = img_overlay_bas.convert("RGBA")
        img_overlay_ann.putalpha(40)
        img_overlay_bas.putalpha(255)


        full_overlay = img_overlay_bas
        # Image.Image.paste(full_overlay, img_overlay_ann)
        full_overlay = Image.blend(full_overlay, img_overlay_ann, 0.25)

        full_overlay_tk = ImageTk.PhotoImage(full_overlay)

        self.canvas.create_image(2*self.corr_width+self.margin*3, self.margin, anchor=tk.NW, image=full_overlay_tk)
        # self.canvas.create_image(2*self.corr_width+20, 0, anchor=tk.NW, image=img_annot_tk)

        self.canvas.image_base=img_base_tk
        self.canvas.image_annot=img_annot_tk
        self.canvas.image_overlay=full_overlay_tk

    
    def setdrawcolor(self, color):
        self.drawcolor=color

    def draw_pixel(self, event):
        color_num_dict = {0:"black", 1:"white"}
        if event.num != "??":
            self.pivot_bit = 1-(int(event.num) >> 1)

        # print(event.num)
        # if event.num == 1:
        if self.pivot_bit:
            self.setdrawcolor("white")
        # if event.num == 3:
        else:
            self.setdrawcolor("black")

        x = event.x // self.pixel_size
        y = event.y // self.pixel_size

        base_bound =    {"upper_bound":self.margin, "lower_bound":self.corr_height+self.margin, "left_bound":self.margin,                  "right_bound":self.corr_width  +self.margin}
        annot_bound =   {"upper_bound":self.margin, "lower_bound":self.corr_height+self.margin, "left_bound":self.margin*2 + self.corr_width,   "right_bound":self.corr_width*2+self.margin*2}
        overlay_bound = {"upper_bound":self.margin, "lower_bound":self.corr_height+self.margin, "left_bound":self.margin*3 + self.corr_width*2, "right_bound":self.corr_width*3+self.margin*3}

        base_bound_corr = dict(zip([key for key in base_bound],[base_bound[key]//self.pixel_size for key in base_bound]))
        annot_bound_corr = dict(zip([key for key in annot_bound],[annot_bound[key]//self.pixel_size for key in annot_bound]))
        overlay_bound_corr = dict(zip([key for key in overlay_bound],[overlay_bound[key]//self.pixel_size for key in overlay_bound]))


        # print(self.drawcolor)

        if self.margin//self.pixel_size <= y < (self.corr_height+self.margin)//self.pixel_size:
            print(f"{x}, {y}")
            # if self.margin <= x < self.width+self.margin:
            if base_bound_corr['left_bound'] <= x < base_bound_corr['right_bound']:
                x1 = (x+self.width) * self.pixel_size + self.margin
                y1 = y * self.pixel_size
                x2 = x1 + self.pixel_size
                y2 = y1 + self.pixel_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.drawcolor, outline=self.drawcolor)
            # if self.width+self.margin*2 <= x < self.width*2+self.margin*2:
            if annot_bound_corr['left_bound'] <= x < annot_bound_corr['right_bound']:
                x1 = x * self.pixel_size
                y1 = y * self.pixel_size
                x2 = x1 + self.pixel_size 
                y2 = y1 + self.pixel_size 
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.drawcolor, outline=self.drawcolor)
            # if self.width*2+self.margin*2 <= x < self.width*3+self.margin*2:
            if overlay_bound_corr['left_bound'] <= x < overlay_bound_corr['right_bound']:
                x1 = (x-self.width) * self.pixel_size - self.margin
                y1 = y * self.pixel_size
                x2 = x1 + self.pixel_size
                y2 = y1 + self.pixel_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.drawcolor, outline=self.drawcolor)
            # self.canvas.create_rectangle(x1+self.width*self.pixel_size, y1, x2+self.width*self.pixel_size, y2, fill=self.drawcolor, outline=self.drawcolor)
            # self.matrix_annotcanvas[y, x-self.width] = self.pivot_bit

root = tk.Tk()
canvas = PixelCanvas(root, 64, 64)
# canvas.placeholder_drawimage("canvas_placeholder_test.png")
canvas.draw_initial_canvas({'base':'test_stomata_viz_BASE.png', 'annot':'test_stomata_viz_ANNOT.png',})
root.mainloop()



# # This takes an image
# # Opens up a canvas window with editing functions
# # Lets you edit it
# # And then close it




# # Functionalities:
# # Lets you draw on any of the three canvases, but only reflects changes on two
# # Black and White drawing only - no opacities
# # Stores the finished annotation to the specific image


# import tkinter as tk
# from PIL import Image, ImageTk

# def display_image(img_path):
#   """Displays the specified image in a Tkinter window."""

#   try:

#         # Create the main window
#     root = tk.Tk()
#     root.title("Image Display")

#     # Open and load the image
#     img = Image.open(img_path)
#     img = img.resize((640, 480))  # Resize for better display (optional)
#     photo = ImageTk.PhotoImage(img)



#     # Create a canvas to display the image
#     canvas = tk.Canvas(root, width=photo.width(), height=photo.height())
#     canvas.pack()

#     # Display the image on the canvas
#     canvas.create_image(0, 0, anchor="nw", image=photo)

#     # Keep the image reference to prevent garbage collection
#     root.image = photo 

#     root.mainloop()

#   except FileNotFoundError:
#     print(f"Error: Image file not found: {img_path}")

# if __name__ == "__main__":
#   img_path = "zeta_maxnoise.png"  # Replace with the actual path to your image
#   display_image(img_path)