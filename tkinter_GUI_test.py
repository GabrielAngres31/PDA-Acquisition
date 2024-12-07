import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
from PIL import Image, ImageTk

class StomataGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Importer")

        self.window_sidelength = 128

        self.canvas_base = tk.Canvas(root, width=self.window_sidelength, height=self.window_sidelength, bg="gray")
        self.canvas_base.grid(row=0, column=0, padx=10, pady=10)

        self.canvas_annot = tk.Canvas(root, width=self.window_sidelength, height=self.window_sidelength, bg="gray")
        self.canvas_annot.grid(row=0, column=1, padx=10, pady=10)

        self.canvas_overlay = tk.Canvas(root, width=self.window_sidelength, height=self.window_sidelength, bg="gray")
        self.canvas_overlay.grid(row=0, column=2, padx=10, pady=10)

        self.button_import_base = tk.Button(root, text="Import Base Image", command=self.import_BASE)
        self.button_import_base.grid(row=1, column=0, padx=10, pady=10)

        self.button_import_annot = tk.Button(root, text="Import Annotation", command=self.import_ANNOT)
        self.button_import_annot.grid(row=1, column=1, padx=10, pady=10)

        self.button_import_csv = tk.Button(root, text="Import CSV", command=self.import_CSV)
        self.button_import_csv.grid(row=1, column=2, padx=10, pady=10)

        self.image_base = None
        self.image_annot = None
        self.image_overlay = None
        self.bbox_coords = [0, 0, self.window_sidelength, self.window_sidelength]
        self.photo = None
        self.bbox_number = tk.IntVar()
        self.bbox_number.set(1)
        self.max_number = 0
        self.df_coords = None
        self.notes_list = []

        self.bbox_entry = tk.Entry(root, textvariable=self.bbox_number, width=4)
        self.bbox_entry.grid(row=2, column=0, padx=3, pady=10, ipadx=0, ipady=0)
        # self.bbox_entry.insert(0, str(self.bbox_number))  # Initialize with current bbox_number
        self.bbox_entry.bind("<Enter>", self.change_value)

        # self.bbox_text_max = tk.Label(root, text = "/")
        # self.bbox_text_max.grid(row=2, column=0, padx=6, pady=10, ipadx=4, ipady=0)

        self.button_decrement = tk.Button(root, text="-", command=self.decrement_bbox)
        self.button_decrement.grid(row=2, column=1, padx=5, pady=10)

        self.button_increment = tk.Button(root, text="+", command=self.increment_bbox)
        self.button_increment.grid(row=2, column=2, padx=5, pady=10)

        self.button_mark_single = tk.Button(root, text="x1")
        self.button_mark_single.grid(row=3, column=0, padx=5, pady=10)

        self.button_mark_double = tk.Button(root, text="x2")
        self.button_mark_double.grid(row=3, column=1, padx=5, pady=10)

        self.button_mark_triple = tk.Button(root, text="x3")
        self.button_mark_triple.grid(row=3, column=2, padx=5, pady=10)

    def x0(self): 
        return -self.bbox_coords[0]

    def y0(self):
        return -self.bbox_coords[1]
    
    def x1(self): 
        return -self.bbox_coords[2]

    def y1(self):
        return -self.bbox_coords[3]

    def xc(self):
        return (self.x0()+self.x1())//2
    
    def yc(self):
        return (self.y0()+self.y1())//2
        
    def xD(self):
        return self.xc()+self.window_sidelength//2
    
    def yD(self):
        return self.yc()+self.window_sidelength//2



    def import_BASE(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif")], initialdir="only_pored/BASE/")
        if file_path:
            self.import_image(self.canvas_base, file_path, 'BASE')
            self.update_overlay()

    def import_ANNOT(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif")], initialdir="only_pored/ANNOT/")
        if file_path:
            self.import_image(self.canvas_annot, file_path, 'ANNOT')
            self.update_overlay()

    def import_CSV(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")], initialdir="inference/")
        if file_path:
            try:

                self.df_coords = pd.read_csv(file_path)
                self.max_number = len(self.df_coords)
                self.bbox_number.set(1) 
                self.notes_list = ["NONE"] * self.max_number
                print("BRUH")
                self.update_bbox_coords(self.df_coords)
                self.update_images()
                self.update_overlay()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read CSV: {e}")

    def update_bbox_coords(self, df_coords):
        if self.bbox_number.get() < len(df_coords) and self.bbox_number.get() >= 1:
            self.bbox_coords = [
                df_coords['bbox-1'][self.bbox_number.get()-1], 
                df_coords['bbox-0'][self.bbox_number.get()-1], 
                df_coords['bbox-3'][self.bbox_number.get()-1], 
                df_coords['bbox-2'][self.bbox_number.get()-1]]
            # self.bbox_entry.delete(0, tk.END)
            # self.bbox_entry.insert(0, str(self.bbox_number))
            # self.bbox_number.set(1)

        else:
            messagebox.showwarning("Warning", "Bounding box number is out of range!")

    def import_image(self, canvas, file_path, image_type):
        img = Image.open(file_path)
        self.update_image(canvas, img, image_type)
        print(f"Imported Image {image_type} to {-self.x0()}, {-self.y0()}")

    def update_image(self, canvas, img, image_type):
        if image_type == 'BASE':
            self.image_base = img
        elif image_type == 'ANNOT':
            self.image_annot = img

        self.photo = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(self.xD(), self.yD(), anchor=tk.NW, image=self.photo)
        canvas.image = self.photo
        print(f"Updated Images to {-self.x0()}, {-self.y0()}")

    def update_overlay(self):
        if self.image_base and self.image_annot:
            result = Image.blend(self.image_base.convert("L"), self.image_annot.convert("L"), alpha=0.37)
            self.image_overlay = ImageTk.PhotoImage(result)
            self.canvas_overlay.delete("all")
            self.canvas_overlay.create_image(self.xD(), self.yD(), anchor=tk.NW, image=self.image_overlay)
            self.canvas_overlay.image = self.image_overlay

    def update_images(self):
        if hasattr(self, 'df_coords') and hasattr(self, 'bbox_number') and self.bbox_number.get() < len(self.df_coords) and self.bbox_number.get() >= 0:
            self.update_bbox_coords(self.df_coords)
            self.update_image(self.canvas_base, self.image_base, "BASE")
            self.update_image(self.canvas_annot, self.image_annot, "ANNOT")
            self.update_overlay()

    def decrement_bbox(self):
        if hasattr(self, 'df_coords') and self.bbox_number.get() > 1:
            s = int(self.bbox_entry.get())
            self.bbox_number.set(s-1)
            self.update_images()

    def increment_bbox(self):
        if hasattr(self, 'df_coords') and self.bbox_number.get() < len(self.df_coords) + 1:
            s = int(self.bbox_entry.get())
            self.bbox_number.set(s+1)
            self.update_images()

    def change_value(self, event):
        placeholder = self.bbox_number.get()
        if hasattr(self, 'df_coords') and self.bbox_number.get() > 0 and self.bbox_number.get() < len(self.df_coords) + 1:
            self.bbox_number.set(self.bbox_entry.get())
            self.update_images()
        else:
            self.bbox_number.set(placeholder)


if __name__ == "__main__":
    root = tk.Tk()
    app = StomataGUI(root)
    root.mainloop()
