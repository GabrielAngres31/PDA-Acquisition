import pandas as pd
import glob
import os

df = pd.DataFrame(columns=["source","replication","treatment","ID","label","bbox-0","bbox-1","bbox-2","bbox-3","area","area_bbox","axis_major_length","axis_minor_length","centroid-0","centroid-1","eccentricity","area_convex","perimeter","equivalent_diameter_area","extent","orientation", "leafsize_um2", "d1", "d2", "p1", "p2"])

handrecord_df = pd.read_csv("C:/Users/Gabriel/Documents/2025_AZD_handrecord.csv")

rowcounter = 0

image_size_dict = dict(
                    zip(
                        [f"R{row.Replicate}-{row.Treatment}-{row.ID[:1]}" for index, row in handrecord_df.iterrows()], [{"leafsize":row["Leaf Size"], "d1":row.d1, "d2":row.d2, "p1":row.p1, "p2":row.p2} for index, row in handrecord_df.iterrows()]
                    )
                )


for file in glob.glob("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/cleaned_images_default/tables/*.csv"):
    # Tricks to get the filename to behave - MUST BE MODIFIED WITH EACH NEW SET OF FILES

    initial_parts = os.path.basename(file).split(".")
    first_part, second_part = initial_parts[0][10:], initial_parts[1][6:]
    first_id_parts  = [i for i in  first_part.split("_")]
    second_id_parts = [j for j in second_part.split("_")]
    replication, treatment, id = first_id_parts[1], first_id_parts[2], second_id_parts[1]
    # print(first_id_parts)
    # print(second_id_parts)
    
    source = f"{replication}-{treatment}-{id}"
    # print(source)

    swap_df = pd.read_csv(file)
    rowcounter += len(swap_df)
    
    swap_df["source"]      = source
    swap_df["replication"] = replication
    swap_df["treatment"]   = treatment
    swap_df["ID"]          = id

    handrecord_put = image_size_dict[source]

    swap_df["leafsize_um2"] = handrecord_put["leafsize"]
    swap_df["d1"] = handrecord_put["d1"]
    swap_df["d2"] = handrecord_put["d2"]
    swap_df["p1"] = handrecord_put["p1"]
    swap_df["p2"] = handrecord_put["p2"]

    df = pd.concat([df, swap_df], ignore_index=True)

assert len(df) == rowcounter, f"rowcounter says {rowcounter}, but the df length is {len(df)}, which is a difference of {len(df)-rowcounter}."
df.to_csv("cleaned_images_default/full_concat_con.csv")

