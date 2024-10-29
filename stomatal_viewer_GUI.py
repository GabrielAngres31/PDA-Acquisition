import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.data

from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import typing
from PIL import Image, ImageTk
from sys import exit


# IMAGE_PATH = "SCD_training_data/source_images/ANNOTATION/OUTLINES/annotations/wt_cot/outline_cot1.tiff"
IMAGE_PATH = "reference_figures/SCANNER_cot1.png"

CSV_PATH = "inference/cot1_STOMATA_MASKS.csv"

CSV_FILE = pd.read_csv(CSV_PATH)
IMAGE_FILE = src.data.load_annotation(IMAGE_PATH)[0]




assert {"bbox-0", "bbox-1", "bbox-2", "bbox-3"}.issubset(CSV_FILE.columns)

bounding_boxes = CSV_FILE[["bbox-0", "bbox-1", "bbox-2", "bbox-3"]]


def get_bbox_from_index(index):
    global bounding_boxes
    BBOX = [c for c in bounding_boxes.iloc[index]]
    return BBOX

def get_subimage(bbox_coords: list):
    PADDING = 32

    midx = (bbox_coords[0]+bbox_coords[2])//2
    midy = (bbox_coords[1]+bbox_coords[3])//2
    upper = midy-PADDING
    lower = midy+PADDING
    left = midx-PADDING
    right = midx+PADDING
    print(midx)
    print(midy)
    subimage = IMAGE_FILE[left:right, upper:lower]
    print(subimage)

    return subimage

def get_upper_left(bbox_coords: list):
    PADDING = 32
    midx = (bbox_coords[0]+bbox_coords[2])//2
    midy = (bbox_coords[1]+bbox_coords[3])//2
    upper = midy-PADDING
    left = midx-PADDING

    return (upper, left)



def load_for_canvas(title: str):
    name = filedialog.askopenfilename(initialdir = "./PDA-Acquisition", title=title)
    if name:
        file = ImageTk.PhotoImage(Image.open(name))
        # return {"file":file, "name":name}
        return file
    return None

def load_image_at_coordinate(file, canvas, coords = [0,0]):
    exec(f"{canvas}.create_image(-{coords[0]}, -{coords[1]}, anchor = NW, image={file})")



root = Tk()
root.geometry("128x384")

window_frame = Frame(master=root)
window_frame.pack(side=TOP)

button_frame = Frame(master=root)
button_frame.pack(side=BOTTOM)

def install_frame(frame_in, frame_name, pack_dir):
    exec(f"global {frame_name}; {frame_name} = Frame(master={frame_in})")
    pack_str = f"side={pack_dir}" if pack_dir else ""
    exec(f"{frame_name}.pack({pack_str}, fill='both')")

def install_canvas(frame_in, canvas_name, bg, size = [64, 64]):
    exec(f"global {frame_in}; {canvas_name} = Canvas({frame_in}, width = {size[0]}, height={size[1]})")
    exec(f"{canvas_name}.pack()")
    exec(f"{canvas_name}.configure(bg='{bg}')")

def install_label(frame_in, label_name, label_text):
    exec(f"global {frame_in}; {label_name} = Label({frame_in}, text = '{label_text}')")
    exec(f"{label_name}.pack()")

def install_button(frame_in, button_name, button_text, command):
    exec(f"global {frame_in}; {button_name} = Button({frame_in}, text = '{button_text}', command = {command})")
    exec(f"{button_name}.pack()")

for frame_list in [
    ["window_frame", "window_base_frame", "TOP"],
    ["window_frame", "window_annot_frame", "TOP"],
    ["window_frame", "window_merge_frame", "TOP"]
]:
    install_frame(*frame_list)

for canvas_list in [
    ["window_base_frame", "base_view", "blue", [64,64]],
    ["window_annot_frame", "annot_view", "orange", [64,64]],
    ["window_merge_frame", "merge_view", "black", [64,64]],
]:
    install_canvas(*canvas_list)

for label_list in [

]:
    install_label(*label_list)

for button_list in [
    ["window_base_frame", "base_load_button", "Base", "lambda: load_image_at_coordinate(load_for_canvas('title'), canvas=window_base_frame)"],
    ["window_annot_frame", "annot_load_button", "Annot", "lambda: print('Hello!')"],
    ["window_merge_frame", "merge_load_button", "Merge", "lambda: print('Hiya!')"],
    ["button_frame", "file_load_button", "Load .csv", "lambda: print('DIE!!!')"]
]:
    install_button(*button_list)

# for frame_list in [
#     ["window_frame",        "window_base_frame",                ""       ],
#     ["window_frame",        "window_annot_frame",               "LEFT"   ],
#     ["window_annot_frame",  "window_annot_clump_frame",         "TOP"    ],
#     ["window_annot_frame",  "window_annot_outline_frame",       "BOTTOM" ],
#     ["window_frame",        "window_overlay_frame",             "RIGHT"  ],
#     ["window_overlay_frame","window_overlay_clump_frame",       "TOP"    ],
#     ["window_overlay_frame","window_overlay_outline_frame",     "BOTTOM" ],
#     ["button_frame",        "button_configure_frame",           "TOP"    ],
#     ["button_frame",        "button_configure_load_frame",      "LEFT"   ],
#     ["button_frame",        "button_configure_checkmark_frame", "RIGHT"  ],
#     ["button_frame",        "button_generator_frame",           "BOTTOM" ]
# ]:
#     install_frame(*frame_list)

# for canvas_list in [
#     ["window_base_frame",           "base_view",            "blue",     [64, 64]],
#     ["window_annot_clump_frame",    "clump_view",           "black",    [64, 64]],
#     ["window_annot_outline_frame",  "outline_view",         "white",    [64, 64]],
#     ["window_overlay_clump_frame",  "overlay_clump_view",   "red",      [64, 64]],
#     ["window_overlay_outline_frame","overlay_outline_view", "green",    [64, 64]]
# ]:
#     install_canvas(*canvas_list)

# for label_list in [
#     ["window_annot_frame",                  "test_A",   "ANNOT"     ],
#     ["window_annot_clump_frame",            "test_Ac",  "clump"     ],
#     ["window_annot_outline_frame",          "test_Ao",  "outline"   ],
#     ["window_base_frame",                   "test_B",   "BASE"      ],
#     ["window_overlay_frame",                "test_O",   "OVERLAY"   ],
#     ["window_overlay_clump_frame",          "test_Oc",  "over_clump"],
#     ["window_overlay_outline_frame",        "test_Oo",  "over_outline"],
#     ["button_configure_frame",              "test_C",   "CONFIGURE" ],
#     ["button_configure_load_frame",         "test_Cl",  "load"      ],
#     ["button_configure_checkmark_frame",    "test_Cc",  "checkmark" ],
#     ["button_generator_frame",              "test_G",   "GENERATOR" ],
# ]:
#     install_label(*label_list)

# for button_list in [
#     ["button_configure_load_frame", "clump_load_button", "Clumps", "lambda: load_image_at_coordinate(load_for_canvas('Select a clump annotation file')['file'], window_annot_clump_frame)"]
# ]:
#     install_button(*button_list)

root.mainloop()
