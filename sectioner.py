# sectioner

import pandas as pd
import PIL.Image
import csv

data = pd.read_csv("only_pored/sections_list.csv")

print(data.iloc[0])

print(len(data))

file_name = ""
file_stuf = ""

for i in range(len(data)):
    row = data.iloc[i]
    if file_name != row["File"]:
        file_name = row["File"]
        file_stuf = PIL.Image.open(f"only_pored/source_images/{file_name}").convert("L")
    fx, fy, dx, dy = row['x'], row['y'], row['x'] + row['size_x'], row['y'] + row['size_y']
    slice = file_stuf.crop((fx, fy, dx, dy))
    slice.save(f"only_pored/BASE/{fx}x_{fy}y__{file_name}")
    
    