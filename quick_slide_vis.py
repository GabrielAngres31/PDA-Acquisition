import matplotlib.pyplot as plt 
import pandas as pd
import os
import os.path
import joypy
import sys

ASSEMBLE_CUMULATIVE_DF = False
if not ASSEMBLE_CUMULATIVE_DF:
    pass
else:

    cumulative_df = pd.DataFrame(columns = ["Unnamed: 0","label","bbox-0","bbox-1","bbox-2","bbox-3","area","axis_major_length","axis_minor_length","eccentricity","area_convex","perimeter","orientation", "dpg", "ID"])

    num_blacklist = [10, 14, 24, 54, 64, 65, 66, 69, 81, 90, 94, 98, 106, 11, 12, 62]

    num_whitelist = [13, 15, 16, 17, 18, 25, 26, 27, 30, 31, 0, 1, 2, 3, 4]

    string_blacklist = [f"{num:04d}" for num in num_blacklist]
    string_whitelist = [f"{num:04d}" for num in num_whitelist]

    print(os.path.basename("inference/match/tables_for_viz/0000_GJA_WT_B1-5dpg-08282024.lif - B1-5dpg-01.tif_e022.output.csv")[15])
    

    # for file in os.listdir("inference/match/tables_for_viz"):
    #     print(os.path.basename(file))
    # files = [[os.path.basename(filename), os.path.basename(filename)[15], os.path.basename(filename)[:4]] for filename in os.listdir("inference/match/tables_for_viz") if not (filename[:4] in string_blacklist) and not (filename == "TOTAL_TABLE.csv") and filename[-4:] == ".csv"]
    for file in os.listdir("inference/match/tables_for_viz"):
        print(os.path.basename(file))
    files = [[os.path.basename(filename), os.path.basename(filename)[15], os.path.basename(filename)[:4]] for filename in os.listdir("inference/match/tables_for_viz") if (filename[:4] in string_whitelist) and not (filename == "TOTAL_TABLE.csv") and filename[-4:] == ".csv"]

    for file in files:
        step_file = pd.read_csv("inference/match/tables_for_viz/"+ file[0])
        step_file['dpg'] = file[1]
        step_file['ID']  = file[2]
        cumulative_df = pd.concat([cumulative_df, step_file])
        
    # cumulative_df.to_csv("inference/match/tables_for_viz/TOTAL_TABLE.csv")
    CONVERSION_pix_to_um = 1.76
    cumulative_df["um_area"] = cumulative_df["area"]/(CONVERSION_pix_to_um**2)
    cumulative_df["um_area_convex"] = cumulative_df["area_convex"]/(CONVERSION_pix_to_um**2)
    cumulative_df["um_axis_major_length"] = cumulative_df["axis_major_length"]/CONVERSION_pix_to_um
    cumulative_df["um_axis_minor_length"] = cumulative_df["axis_minor_length"]/CONVERSION_pix_to_um
    cumulative_df["um_perimeter"] = cumulative_df["perimeter"]/CONVERSION_pix_to_um

    cumulative_df["aspect_ratio"] = cumulative_df["um_axis_major_length"]/cumulative_df["um_axis_minor_length"]
    cumulative_df.to_csv("inference/match/tables_for_viz/TOTAL_TABLE_combined.csv")
    #print(cumulative_df.head())





# df = pd.read_csv("inference/match/tables_for_viz/TOTAL_TABLE.csv")
df = pd.read_csv("inference/match/tables_for_viz/TOTAL_TABLE_combined.csv")
#print(args.source_data)
#print(df)

df_3 = df[df.dpg == 3]
df_4 = df[df.dpg == 4]
df_5 = df[df.dpg == 5]
df_7 = df[df.dpg == 7]

df_lists = [[df_3, 3], [df_4, 4], [df_5, 5], [df_7, 7]]
measures = [["um_area", (0, 1350)], ["um_axis_major_length", (0, 45)], ["um_axis_minor_length", (0, 45)], ["eccentricity", (0, 1)], ["um_area_convex", (0, 1350)], ["um_perimeter", (0, 170)], ["orientation", (-3, 3)], ["aspect_ratio", (0.8, 1.8)]]
measure_pairs = [["um_axis_major_length", "um_area_convex"], ["um_axis_major_length", "um_axis_minor_length"], ["um_axis_major_length", "orientation"]]


# for df_list in df_lists:

#     df_work = df_list[0]
#     for measure in measures:
#         # plt.hist(clump_info_dict.values(), bins=list(range(0, 1000, 50)))
#         # plt.title("Clump Sizes")
#         # plt.show()
#         #print(clump_info_dict)
#         #assert 'area' in table.keys(), "Area is not listed in this table!"
#         plt.figure(figsize = (8,6))
#         plt.hist(df_list[0][measure[0]])#, bins=list(range(0, 2000, 120)))
#         plt.title(f"{measure[0]}")
#         plt.savefig(f"inference/match/tables_for_viz/graphs/{measure[0]}_{df_list[1]}dpg.png")
#         plt.clf()
# # if args.scatterplots:
# #     assert not ("ID" in df), "Too many groups!"
# #     plots=args.scatterplots.split("|")
# #     for plot in plots:
# #         #print(plot)
# #         x,y = plot.split(",")
# #         plt.figure(figsize = (8,6))
# #         plt.scatter(df[x], df[y])
# #         plt.axis(xmin=0, ymin=0, xmax=args.xmax, ymax=args.ymax)
# #         plt.title(f"{x} vs. {y}")
# #         plt.savefig(f"reference_figures/visualizers_test/{args.save_as}_{x}_vs_{y}_scatter.png")
# #         plt.clf()



# #print(df.head())


#     dpg_mapping = {1:"#fde725FF", 
#                 2:"#90d743FF", 
#                 3:"#35b779FF", 
#                 4:"#21918cFF", 
#                 5:"#31688eFF", 
#                 6:"#443983FF", 
#                 7:"#440154FF"}

#     sorted_count_by_id = df_work.groupby("ID").size().sort_values(ascending=True)
#     sortcount_to_dict = dict(zip([k for k in sorted_count_by_id.keys()], 
#                                     [v for v in sorted_count_by_id.values]))

#     df_work["num"] = df_work.apply(lambda row: sortcount_to_dict[row["ID"]], axis=1)
#     #print(df.head())
#     grouped_by_ID = df_work.groupby("ID", sort=False)
#     # NOTE: this assumes that all items in a group have the same dpg
#     dpg_by_ID    = [ list(group['dpg'])[0]  for _, group in grouped_by_ID]
#     colors_by_ID = [ dpg_mapping[dpg] for dpg in dpg_by_ID ]

#     for plot in measures:
#         fig, ax = joypy.joyplot(df_work.groupby("ID", sort=False), by = "ID", column = plot[0], x_range=plot[1], fade = True, color = colors_by_ID)
#         plt.title(f"Ridgeplot of {plot[0]} at {df_list[1]}dpg")
#         plt.savefig(f"inference/match/tables_for_viz/graphs/ridge_{plot[0]}_{df_list[1]}dpg.png", bbox_inches = "tight")
#         plt.clf()
#     df_work = df_work.sort_values("num", ascending=True)
#     print(df_work.head())
#     # for plot in measures:
#     #     # fig, ax = joypy.joyplot(df, by="num", column = plot, fade = True)
#     #     fig, ax = joypy.joyplot(df.groupby("ID", sort=False), column = plot, fade = True, color = colors_by_ID)
#     #     plt.title(f"Ridgeplot of {plot}")
#     #     plt.savefig(f"reference_figures/visualizers_test/{args.save_as}_{plot}_num_ridgeplot.png", bbox_inches = "tight")
#     #     plt.clf()
#     # print(str(grouped_by_ID.size()))
#     # print(str(grouped_by_ID.size().sort_values(ascending=True)))

#     colors_by_norm = []

# plt.show()


# return True


# search through "match_filter" using the numbers in [blacklist]
# get starting index of source file string

dpg_mapping = {1:"#fde725FF", 
                2:"#90d743FF", 
                3:"#35b779FF", 
                4:"#21918cFF", 
                5:"#31688eFF", 
                6:"#443983FF", 
                7:"#440154FF"}

sorted_count_by_id = df.groupby("ID").size().sort_values(ascending=True)
sortcount_to_dict = dict(zip([k for k in sorted_count_by_id.keys()], 
                                [v for v in sorted_count_by_id.values]))

df["num"] = df.apply(lambda row: sortcount_to_dict[row["ID"]], axis=1)
#print(df.head())
grouped_by_ID = df.groupby("ID", sort=False)
grouped_by_dpg = df.groupby("dpg", sort = True)
# NOTE: this assumes that all items in a group have the same dpg
dpg_by_ID    = [ list(group['dpg'])[0]  for _, group in grouped_by_ID]
colors_by_ID = [ dpg_mapping[dpg] for dpg in dpg_by_ID ]

for plot in measures:
    fig, ax = joypy.joyplot(df.groupby("ID", sort=False), by = "dpg", column = plot[0], x_range=plot[1], fade = True, color = colors_by_ID)
    if plot[0] in ["um_area", "um_area_convex"]:
        plt.title(f"Ridgeplot of {plot[0]} (um^2)")
    else:
        plt.title(f"Ridgeplot of {plot[0]} (um)")
    plt.savefig(f"inference/match/tables_for_viz/graphs/ridge_{plot[0]}_combined.png", bbox_inches = "tight")
    plt.clf()
    print(f"Finished with {plot[0]}")