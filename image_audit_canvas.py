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
    def __init__(self, master, width, height, base_section=None, annot_section=None, pixel_size=7):
        self.master = master
        self.width = width
        self.height = height
        self.pixel_size = pixel_size

        self.matrix_basecanvas = np.zeros((height, width), dtype=int)
        self.matrix_annotcanvas = np.zeros((height, width), dtype=int)
        self.matrix_overlaycanvas = np.zeros((height, width), dtype=int)

        self.testimage = ImageTk.PhotoImage(Image.open("zeta_maxnoise.png"))
        
        self.drawcolor = "white"

        self.canvas = tk.Canvas(master, width=width * pixel_size, height=height * pixel_size, bg="black", cursor="plus")
        self.canvas.pack()
      
        self.canvas.image = self.testimage

        self.canvas.bind("<Button-1>", self.draw_pixel)
        self.canvas.bind("<B1-Motion>", self.draw_pixel)
        self.canvas.bind("<ButtonRelease-1>", self.draw_pixel)

        self.canvas.bind("<Button-3>", self.draw_pixel)
        self.canvas.bind("<B3-Motion>", self.draw_pixel)
        self.canvas.bind("<ButtonRelease-3>", self.draw_pixel)

    def placeholder_drawimage(self, imgpath):
        if os.path.exists(imgpath):
            img = ImageTk.PhotoImage(Image.open(imgpath))
            img_placeholder = self.canvas.create_image(0, 0, image=img)
            self.canvas.image = img
        else:
            print(f"Error: Image file not found: {imgpath}")

    
    def setdrawcolor(self, color):
        self.drawcolor=color

    def draw_pixel(self, event):

        # print(event.num)
        if event.num == 1:
            self.setdrawcolor("white")
        if event.num == 3:
            self.setdrawcolor("black")

        x = event.x // self.pixel_size
        y = event.y // self.pixel_size
        # print(self.drawcolor)

        if 0 <= x < self.width and 0 <= y < self.height:
            x1 = x * self.pixel_size
            y1 = y * self.pixel_size
            x2 = x1 + self.pixel_size
            y2 = y1 + self.pixel_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.drawcolor, outline=self.drawcolor)
            self.matrix_annotcanvas[y, x]



root = tk.Tk()
canvas = PixelCanvas(root, 64, 64)
canvas.placeholder_drawimage("zeta_maxnoise.png")
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