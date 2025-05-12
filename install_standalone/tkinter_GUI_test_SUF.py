# Import necessary packages

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from PIL import Image, ImageTk
import os
import image_audit_canvas_SUF_2
import subprocess



class StomataGUI:
    def __init__(self, root):

        # Load Config options
        ### Read config file
        ### Loads: Recent BASE, ANNOT, and CSV files
        self.configFilePath = "annotation_helper_files/stomata_gui_config.txt"
         
        with open(self.configFilePath, 'r') as file:
            lines = file.readlines()
            self.config_properties = dict([(v for v in line.strip().split("=")) for line in lines])

        file_to_dir = dict(zip(['recent_BASE', 'recent_ANNOT', 'recent_CSV'], ['dir_BASE', 'dir_ANNOT', 'dir_CSV']))

        for prop_file in ['recent_BASE', 
                     'recent_ANNOT',
                     'recent_CSV']:
            # Check that all the filepaths from the last session are still valid
            # This can change if files have been deleted or moved
            try: 
                assert os.path.exists(self.config_properties[prop_file])
            # If a path isn't available, the user is prompted to select a new file to load.
            # TODO: Make Warning Window contain informative text
            except:
                messagebox.showwarning("Missing File")
                file_set = filedialog.askopenfilename(title=f"Select File: {prop_file}")
                self.config_properties[prop_file] = file_set
                self.config_properties[file_to_dir[prop_file]] = os.path.dirname(file_set)
            
        
        for prop_dir in ['dir_BASE',
                     'dir_ANNOT', 
                     'dir_CSV']:
            # Check that file directories to select files from are still valid.
            # File directories may be changed or deleted.
            # If any of the files above were missing and reselected through user prompt, 
            #   it automatically updates the directory here, so it shouldn't trigger an error.
            # Actually, if you have a directory error without having a file error, then something really bizarre must be going on.
            try: 
                assert os.path.exists(self.config_properties[prop_dir])
                # print("bround")
            except:
                messagebox.showwarning("Missing Directory")
                filedialog.askopenfilename(title=f"Select Directory: {prop_dir}")


        # Window Setup
        self.root = root
        self.root.title("Image Importer")

        # Navigable Tabs
        self.main_tab_control = ttk.Notebook(root)
        self.inference_tab = ttk.Frame(self.main_tab_control)
        self.image_compare_tab = ttk.Frame(self.main_tab_control)


        ### Menu Bars
        self.menubar = tk.Menu(root)
        self.root.config(menu = self.menubar)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.editmenu = tk.Menu(self.menubar, tearoff=0)
        self.manamenu = tk.Menu(self.menubar, tearoff=0)

        self.main_tab_control.add(self.image_compare_tab, text='Annotator')
        self.main_tab_control.add(self.inference_tab, text='Inference (WIP)')

        self.main_tab_control.pack(expand=1, fill="both")

        def donothing(): l=0

        ### "File" Menu Options                
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.menubar.add_cascade(label="Edit", menu=self.editmenu)
        self.menubar.add_cascade(label="Manage", menu=self.manamenu)

        self.filemenu.add_command(label="Load Base", command=self.import_BASE_dialog)
        self.filemenu.add_command(label="Load Annot", command=self.import_ANNOT_dialog)
        self.filemenu.add_command(label="Load CSV", command=self.import_CSV_dialog)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=root.quit)
        
        ### "Edit" Menu Options     
        self.editmenu.add_command(label="Edit Annotation...", command=self.open_window_edit)
        self.root.bind("E", self.open_window_edit)
        self.editmenu.add_command(label="Confirm Annotation", command=self.confirm_annot) 
        self.root.bind("<Shift_L>", lambda event: self.confirm_annot)
        self.editmenu.add_command(label="Confirm Notes", command=self.confirm_notes) 
        self.root.bind("<Alt-L>", lambda event: self.confirm_notes(event=event))
        self.editmenu.add_command(label="Clear Note", command=self.clear_notes)

        # print(self.root.bind())

        ### Manager Menu Options
        self.manamenu.add_command(label="Paired Files (WIP)", command=donothing)
        self.manamenu.add_separator()
        self.manamenu.add_command(label="Set Recents", command=self.set_recents)
        self.manamenu.add_command(label="Set Paired Files (WIP)", command=self.set_paired_files)

        # Hardcoded window sidelength
        self.window_sidelength = 148

        # Set up FileTitle
        self.file_label = tk.Label(self.image_compare_tab, text=os.path.basename(self.config_properties["recent_BASE"]))
        self.file_label.grid(row=0, column=1, padx=10, pady=8)

        # Set up canvases
        self.canvas_base = tk.Canvas(self.image_compare_tab, width=self.window_sidelength, height=self.window_sidelength, bg="gray")
        self.canvas_base.grid(row=1, column=0, padx=10, pady=8)

        self.canvas_annot = tk.Canvas(self.image_compare_tab, width=self.window_sidelength, height=self.window_sidelength, bg="gray")
        self.canvas_annot.grid(row=1, column=1, padx=10, pady=8)

        self.canvas_overlay = tk.Canvas(self.image_compare_tab, width=self.window_sidelength, height=self.window_sidelength, bg="gray")
        self.canvas_overlay.grid(row=1, column=2, padx=10, pady=8)

        # Set up Buttons
        self.button_import_base = tk.Button(self.image_compare_tab, text="Import Base Image", command=self.import_BASE_dialog)
        self.button_import_base.grid(row=2, column=0, padx=10, pady=10)

        self.button_import_annot = tk.Button(self.image_compare_tab, text="Import Annotation", command=self.import_ANNOT_dialog)
        self.button_import_annot.grid(row=2, column=1, padx=10, pady=10)

        self.button_import_csv = tk.Button(self.image_compare_tab, text="Import CSV", command=self.import_CSV_dialog)
        self.button_import_csv.grid(row=2, column=2, padx=10, pady=10)

        ### Setup Inference Tab

        # Set up Buttons
        self.button_set_test = tk.Button(self.inference_tab, text="Test Split", command=self.set_test_split)
        self.button_set_test.grid(row=0, column=0, padx=10, pady=10)

        self.button_set_val = tk.Button(self.inference_tab, text="Val Split", command=self.set_val_split)
        self.button_set_val.grid(row=0, column=1, padx=10, pady=10)

        self.button_checkfiles = tk.Button(self.inference_tab, text="Check Files", command=self.verify_split_files)
        self.button_checkfiles.grid(row=1, column=0, padx=10, pady=10)

        self.button_run_inference = tk.Button(self.inference_tab, text="Run Inference", command=self.run_inference)
        self.button_run_inference.grid(row=1, column=1, padx=10, pady=10)

        # Setup Variables
        self.image_base = None
        self.image_annot = None
        self.image_overlay = None

        self.current_annot_path = ""

        self.bbox_coords = [0, 0, self.window_sidelength, self.window_sidelength]
        self.photo = None # Placeholder for image updating
        self.bbox_number = tk.IntVar()
        self.bbox_number.set(1)
        self.max_number = 0 # How many clumps there are
        self.df_coords = None
        self.notes_list = []

        self.confirm_annot_num = 0
        self.confirm_annot_bbox_coords = [0, 0, 0, 0]
        self.confirm_annot_corner = [0, 0]

        self.opacity_lower_bound = 0
        
        self.advance_on_label=tk.IntVar()

        # Clump ID Entry
        self.bbox_entry = tk.Entry(self.image_compare_tab, textvariable=self.bbox_number, width=4)
        self.bbox_entry.grid(row=3, column=0, padx=3, pady=10, ipadx=0, ipady=0)

        # self.bbox_entry.bind("<Enter>", self.change_value)

        # +/- (>>/<<) buttons
        self.button_decrement = tk.Button(self.image_compare_tab, text="<<", command=self.decrement_bbox)
        self.root.bind("q", self.decrement_bbox) 
        self.button_decrement.grid(row=3, column=1, padx=5, pady=10)

        self.button_increment = tk.Button(self.image_compare_tab, text=">>", command=self.increment_bbox)
        self.root.bind("w", self.increment_bbox)
        self.button_increment.grid(row=3, column=2, padx=5, pady=10)

        # Advance checkbox
        self.checkbox_advance = tk.Checkbutton(self.image_compare_tab, text="Advance on Label", variable=self.advance_on_label, command=lambda:print(self.advance_on_label.get()))
        self.checkbox_advance.grid(row=4, column=1, padx=5, pady=10)

        # Current Label
        self.current_labels = tk.Label(self.image_compare_tab, text=0)
        self.current_labels.grid(row=4, column=0, padx=5, pady=10)

        # Label Buttons
        self.button_mark_edge = tk.Button(self.image_compare_tab, text="Edge", command=lambda: self.mark_note("Edge"))
        self.image_compare_tab.bind("a", lambda event: self.mark_note("Edge"))
        self.button_mark_edge.grid(row=5, column=0, padx=5, pady=10)
        # self.button_mark_edge = tk.Button(image_compare_tab, text="Edge", command=lambda: self.mark_note("Edge"))
        # image_compare_tab.bind("a", lambda event: self.mark_note("Edge"))
        # self.button_mark_edge.grid(row=5, column=0, padx=5, pady=10)

        self.button_mark_cluster = tk.Button(self.image_compare_tab, text="Cluster", command=lambda: self.mark_note("Cluster"))
        self.root.bind("s", lambda event: self.mark_note("Cluster")) 
        self.button_mark_cluster.grid(row=5, column=1, padx=5, pady=10)

        self.button_mark_error = tk.Button(self.image_compare_tab, text="ERROR", command=lambda: self.mark_note("ERROR"))
        self.root.bind("d", lambda event: self.mark_note("ERROR"))        
        self.button_mark_error.grid(row=5, column=2, padx=5, pady=10)

        self.button_mark_pore = tk.Button(self.image_compare_tab, text="Pore", command=lambda: self.mark_note("Pore"))
        self.root.bind("z", lambda event: self.mark_note("Pore"))
        self.button_mark_pore.grid(row=6, column=0, padx=5, pady=10)

        self.button_mark_unknown = tk.Button(self.image_compare_tab, text="Unsure", command=lambda: self.mark_note("Unsure"))
        self.root.bind("x", lambda event: self.mark_note("Unsure"))
        self.button_mark_unknown.grid(row=6, column=1, padx=5, pady=10)
        
        self.button_mark_nopore = tk.Button(self.image_compare_tab, text="No Pore", command=lambda: self.mark_note("NoPore"))
        self.root.bind("c", lambda event: self.mark_note("NoPore"))
        self.button_mark_nopore.grid(row=6, column=2, padx=5, pady=10)

        self.root.bind("<Control_L>", self.clear_notes)

        self.main_tab_control.bind('<<NotebookTabChanged>>', self.set_focus_to_tab)

        if "recent_BASE" in self.config_properties and self.config_properties["recent_BASE"]:
            self.import_BASE(self.config_properties["recent_BASE"])
        if "recent_ANNOT" in self.config_properties and self.config_properties["recent_ANNOT"]:
            self.import_ANNOT(self.config_properties["recent_ANNOT"])
        if "recent_CSV" in self.config_properties and self.config_properties["recent_CSV"]:            
            self.import_CSV(self.config_properties["recent_CSV"])

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # self.set_focus_to_tab(None, 0)
        self.main_tab_control.select(0)
        # self.set_focus_to_tab(None)

    def set_focus_to_tab(self, event=None):
        ind = self.main_tab_control.index(self.main_tab_control.select())
        # ind = self.main_tab_control.index(self.main_tab_control.select())
        print("None")

        # self.main_tab_control.focus_set(ind)

    
        print("Hm?")
        # self.main_tab_control.select(tab_index)
        print(self.main_tab_control.index(self.main_tab_control.select()))
        
        self.main_tab_control.select(ind)
        print(ind)
        print(self.root.focus_get())


    # Set config properties during function calls
    def set_property(self, property, value):
        self.config_properties[property] = value

    # Upper left corner of clump bounding box
    def x0(self): 
        return -self.bbox_coords[0]

    def y0(self):
        return -self.bbox_coords[1]
    
    # Low right corner of clump bounding box
    def x1(self): 
        return -self.bbox_coords[2]

    def y1(self):
        return -self.bbox_coords[3]

    # Center pixel
    def xc(self):
        return (self.x0()+self.x1())//2
    
    def yc(self):
        return (self.y0()+self.y1())//2
    
    # Fixed-size window centered on chunk, upper left corner
    def xD(self):
        return self.xc()+self.window_sidelength//2
    
    def yD(self):
        return self.yc()+self.window_sidelength//2
    
    # File Dialogs
    ### Get the BASE file (dialog and autofunction)
    def import_BASE_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif")], initialdir=self.config_properties['dir_BASE'])
        if file_path: self.import_BASE(file_path)
    def import_BASE(self, file_path):
        self.import_image(self.canvas_base, file_path, 'BASE')
        self.update_overlay()
        self.set_property("recent_BASE", file_path)
        self.set_property("dir_BASE", os.path.dirname(file_path))
        self.file_label.config(text=os.path.basename(file_path))

    ### Get the ANNOT file (dialog and autofunction)
    def import_ANNOT_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif")], initialdir=self.config_properties['dir_ANNOT'])
        if file_path:
            self.import_ANNOT(file_path)
            # Putting this here to prevent losing all your work on an accidental file switch
            # THIS DOESN'T WORK!!!!
            self.image_annot.save(os.path.abspath(rf'{self.config_properties["recent_ANNOT"]}'))
            print("Saving work on ANNOT!")

    def import_ANNOT(self, file_path):
        self.import_image(self.canvas_annot, file_path, 'ANNOT')
        self.update_overlay()
        self.current_annot_path = file_path
        self.set_property("recent_ANNOT", file_path)
        self.set_property("dir_ANNOT", os.path.dirname(file_path))

    ### Get the CSV file (dialog and autofunction)
    def import_CSV_dialog(self):
        decision = messagebox.askyesnocancel('Get Annotation File', 'Use premade clumps file?')
        if decision:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")], initialdir=self.config_properties['dir_CSV'])
        elif not decision:
            check = messagebox.askyesnocancel('Confirm Annotation', 'Do you have the correct file to make a clumps list of?')
            if check:
                print("Generating clumps file")
                subprocess.run(f'python.exe clumps_table_SUF.py --input_path="{self.config_properties["recent_ANNOT"]}" --output_folder="annotation_helper_files/"', shell=True) #, capture_output=True)
                # file_path = f"annotation_helper_files/{os.path.splitext(os.path.basename(self.config_properties['recent_ANNOT']))[0]}.csv"
                assert os.path.exists(self.config_properties['recent_ANNOT'])
                # print(os.path.basename(self.config_properties['recent_ANNOT']))
                # print(os.path.splitext(os.path.basename(self.config_properties['recent_ANNOT'])))
                # file_path = f"annotation_helper_files/{os.path.splitext(os.path.basename(self.config_properties['recent_ANNOT']))[0]}.csv"
                # file_path = f"annotation_helper_files/{os.path.splitext(os.path.basename(self.config_properties['recent_ANNOT']))[0]}.csv"
        if file_path:
            print("did you get here already?")
            try:
                self.import_CSV(file_path)
                print("Successfully loaded csv!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read CSV: {e}")
    def import_CSV(self, file_path):
        # print(f"{file_path}")
        self.df_coords = pd.read_csv(file_path, encoding='utf8')
        self.max_number = len(self.df_coords)
        self.bbox_number.set(1) 
        try: 
            self.notes_list = self.df_coords["Notes"]
        except:
            print("Writing Notes Column: ")
            self.notes_list = ["NONE"] * self.max_number
        self.update_bbox_coords(self.df_coords)
        self.update_images()
        self.update_overlay()
        self.set_property("recent_CSV", file_path)
        self.set_property("dir_CSV", os.path.dirname(file_path))

    # Change current bbox coords for focused clump
    def update_bbox_coords(self, df_coords):
        if self.bbox_number.get() < len(df_coords)+1 and self.bbox_number.get() >= 1:
            self.bbox_coords = [
                df_coords['bbox-1'][self.bbox_number.get()-1], 
                df_coords['bbox-0'][self.bbox_number.get()-1], 
                df_coords['bbox-3'][self.bbox_number.get()-1], 
                df_coords['bbox-2'][self.bbox_number.get()-1]]
            self.current_labels.config(text=self.notes_list[self.bbox_number.get()-1])
            # print(self.bbox_number.get())
        else:
            messagebox.showwarning("Warning", "Bounding box number is out of range!")

    # Import an image for display
    def import_image(self, canvas, file_path, image_type):
        try:
            img = Image.open(file_path)
            self.update_image(canvas, img, image_type)
            print(f"Imported Image {image_type} to {-self.x0()}, {-self.y0()}")
        except:
            print(f"Could not import image. Check your filepath: {file_path}")
        
    # Change image on a given canvas
    def update_image(self, canvas, img, image_type):
        if image_type == 'BASE':
            self.image_base = img
        elif image_type == 'ANNOT':
            self.image_annot = img


        self.photo = ImageTk.PhotoImage(img)
        # img.show()
        canvas.delete("all")
        canvas.create_image(self.xD(), self.yD(), anchor=tk.NW, image=self.photo)
        # print(self.xc(), self.yc())
        canvas.image = self.photo
        # print(f"Updated Images to {-self.x0()}, {-self.y0()}")

    # Change the computed overlay
    def update_overlay(self):
        if self.image_base and self.image_annot:
            try:
                assert self.image_base.size == self.image_annot.size, f"There's an image shape mismatch! Base: {self.image_base.size} || Annot: {self.image_annot.size}"
                result = Image.blend(self.image_base.convert("L"), self.image_annot.convert("L"), alpha=0.37)
                self.image_overlay = ImageTk.PhotoImage(result)
                self.canvas_overlay.delete("all")
                self.canvas_overlay.create_image(self.xD(), self.yD(), anchor=tk.NW, image=self.image_overlay)
                self.canvas_overlay.image = self.image_overlay
            except: 
                print(f"Image dimension mismatch or other problem detected. Clearing Canvas.\n----Image Dimensions are {self.image_base.size} vs. {self.image_annot.size}")
                self.canvas_overlay.delete("all")

    # Update BASE, ANNOT, and OVERLAY on bbox change
    def update_images(self):
        if hasattr(self, 'df_coords') and hasattr(self, 'bbox_number') and self.bbox_number.get() < len(self.df_coords)+1 and self.bbox_number.get() >= 0:
            self.update_bbox_coords(self.df_coords)
            self.update_image(self.canvas_base, self.image_base, "BASE")
            self.update_image(self.canvas_annot, self.image_annot, "ANNOT")
            self.update_overlay()

    # Go one clump back
    def decrement_bbox(self, event=None):
        if hasattr(self, 'df_coords') and self.bbox_number.get() > 1:
            # print("-1")
            self.confirm_annot()
            s = int(self.bbox_entry.get())
            self.bbox_number.set(s-1)
            
            self.update_images()

    # Go one clump forward
    def increment_bbox(self, event=None):
        if hasattr(self, 'df_coords') and self.bbox_number.get() < len(self.df_coords):
            self.confirm_annot()
            s = int(self.bbox_entry.get())
            self.bbox_number.set(s+1)

            self.update_images()

    # Move to an arbitrary clump (user input) (CURRENTLY UNUSED)
    def change_value(self, event):
        placeholder = self.bbox_number.get()
        if hasattr(self, 'df_coords') and self.bbox_number.get() > 0 and self.bbox_number.get() < len(self.df_coords) + 1:
            self.confirm_annot()
            self.bbox_number.set(self.bbox_entry.get())
            self.update_images()
        else:
            self.bbox_number.set(placeholder)
        root.focus_set()
    
    # Set recent files to current workspace files
    def set_recents(self):
        with open(self.configFilePath, 'w') as file:
            for k in self.config_properties:
                file.write(f"{k}={self.config_properties[k]}\n")
    
    # Write annotation notes to CSV file
    def update_csv_notes(self):
        write_path = self.config_properties['recent_CSV']
        self.df_coords
        self.df_coords.to_csv(write_path)

    # Lets you associate three files (BASE, ANNOT, CSV) together so you can load them with a single click. Does nothing right now.
    def set_paired_files(self):
        #TODO
        pass

    def mark_note(self, note_text):
        index = self.bbox_number.get()-1
        try:
            if self.notes_list[index] == "NONE": 
                self.notes_list[index] = note_text
            elif note_text in self.notes_list[index]:
                pass
            else:
                self.notes_list[index] = self.notes_list[index] + f", {note_text}"
            if self.advance_on_label.get():
                self.increment_bbox()
        except:
            print("Something went wrong while trying to __SET the label__! Hopefully nothing got broken...")
        
        # TODO: write to self.notes_list
        # On window close, write self.notes_list to the file!
    
    def clear_notes(self, event=None):
        index = self.bbox_number.get()-1
        try:
            if self.notes_list[index] != "NONE": 
                # print(self.notes_list[index])
                self.notes_list[index] = "NONE"
            if self.advance_on_label.get():
                self.increment_bbox()
        except:
            print("Something went wrong while trying to __CLEAR the label__! Hopefully nothing got broken...")

    def note_summary_stats(self):
        # for item in list(set(self.notes_list)):
        #     print(list(set(self.notes_list)))
        #     print(item)
        #     print(self.notes_list.count(item))
        unique_items = {}
        for item in self.notes_list:
            if item in unique_items:
                unique_items[item] += 1
            else:
                unique_items[item] = 1
        # print(unique_items)

    def open_window_edit(self, event=None):
        window_size = 64
        pixel_size = 6
        global crop_coords
        crop_coords = (-(self.xc()+window_size//2), -(self.yc()+window_size//2), -(self.xc()-window_size//2), -(self.yc()-window_size//2))
        # print(crop_coords)
        save_base = self.image_base.crop(crop_coords)
        save_annot = self.image_annot.crop(crop_coords)
        
        # Get the two images
        
        self.confirm_annot_num = self.bbox_number
        self.confirm_annot_bbox_coords = self.bbox_coords
        self.confirm_annot_corner = [-self.xc(), -self.yc()]
        
        
        # Save the two images to that folder with defined names
        save_base.convert("L").save("annotation_helper_files/save_base_file.png")
        
        save_annot.convert("L").save("annotation_helper_files/save_annot_file.png")
        # Pass the paths as an arg to the subprocess command
        edit_root = tk.Toplevel(self.root)
        canvas = image_audit_canvas_SUF_2.PixelCanvas(edit_root, window_size, window_size, base_section_path="annotation_helper_files/save_base_file.png", annot_section_path="annotation_helper_files/save_annot_file.png", pixel_size=pixel_size)

        
        # canvas.protocol("WM_DELETE_WINDOW", canvas.on_closing)
        # subprocess.run("python image_audit_canvas.py --base_path=annotation_helper_files/save_base_file.png --annot_path=annotation_helper_files/save_annot_file.jpg", shell=True)        
    
    def confirm_annot(self):
        try:
            global crop_coords
            img_repaste = Image.open("annotation_helper_files/save_annot_file.png")
            # img_repaste.show()
            img_repaste = img_repaste.convert("L")
            # print(img_repaste.mode)
            # img_to_save = self.image_annot.copy()
            # img_to_save.paste(img_repaste, box=(crop_coords[0], crop_coords[1]))
            # self.image_annot.mode = "L"
            self.image_annot = self.image_annot.convert("L")
            self.image_annot.paste(img_repaste, (crop_coords[0], crop_coords[1]))
            # Image.Image.paste(img_repaste, (crop_coords[0], crop_coords[1], crop_coords[0]+64, crop_coords[1]+64), self.image_annot)
            # print([crop_coords[0], crop_coords[1]])
            # self.image_annot.show()3
        except NameError:
            pass

    def confirm_notes(self, event=None):
        # self.image_annot.save(self.config_properties["recent_ANNOT"])
        self.df_coords["Notes"] = self.notes_list
        self.update_csv_notes()
        path = self.config_properties["recent_ANNOT"]
        print(f"--| Confirmed Notes for {os.path.basename(path)}")
    
    def set_test_split(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")], initialdir=self.config_properties['dir_test_split'])
        if file_path:
            return file_path
    def set_val_split(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")], initialdir=self.config_properties['dir_val_split'])
        if file_path:
            return file_path
    def verify_split_files(self):
        pass
    def run_inference(self):
        pth_path = filedialog.askopenfilename(filetypes=[("PTH Files", "*.PTH")], initialdir="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/checkpoints")
        try:
            assert pth_path
        except:
            print("You must select weights to run an inference!")
        inf_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif")], initialdir=self.config_properties['dir_BASE'])
        try:
            assert inf_path
        except:
            print("You need to run inference on an image. Please select an image.")
        subprocess.run(f'python inference.py --model={pth_path} --input="{inf_path}" --overlap=128 --outputname="recent_GUI_output"', shell=True)
        pass

    def on_closing(self):
        print("Window is closing...")
        self.df_coords["Notes"] = self.notes_list
        self.update_csv_notes()
        self.note_summary_stats()
        self.set_recents()
        print("something's happening")
        assert os.path.exists(self.config_properties["recent_ANNOT"])
        assert os.path.exists(self.config_properties["recent_CSV"])
        # print(self.config_properties["recent_ANNOT"])f
        self.image_annot.save(os.path.abspath(rf'{self.config_properties["recent_ANNOT"]}'))
        print("Just ran the save image command")
        # self.image_annot.show()
        self.root.destroy()  # Close the window



if __name__ == "__main__": #
    root = tk.Tk()
    app = StomataGUI(root)
    root.mainloop()


# WHERE IS YOUR LABEL :gun:

# AUTOMATIC CLUMPFINDING ON ANNOT LOAD