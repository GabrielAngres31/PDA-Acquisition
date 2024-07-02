# visualizers
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib
import pandas as pd
import numpy as np
import joypy
import argparse
import torch
import os

def main(args:argparse.Namespace) -> bool:
    df = pd.read_csv(args.source_data)
    print(args.source_data)
    print(df)

    if args.histograms:
        assert not ("ID" in df), "Too many groups!"
        measures = args.histograms.split(",")
        for measure in measures:
            # plt.hist(clump_info_dict.values(), bins=list(range(0, 1000, 50)))
            # plt.title("Clump Sizes")
            # plt.show()
            #print(clump_info_dict)
            #assert 'area' in table.keys(), "Area is not listed in this table!"
            plt.figure(figsize = (8,6))
            plt.hist(df[measure])#, bins=list(range(0, 2000, 120)))
            plt.title(f"{measure}")
            plt.savefig(f"reference_figures/visualizers_test/test_{measure}_histogram_cot6.png")
            plt.clf()
    if args.scatterplots:
        assert not ("ID" in df), "Too many groups!"
        plots=args.scatterplots.split("|")
        for plot in plots:
            print(plot)
            x,y = plot.split(",")
            plt.figure(figsize = (8,6))
            plt.scatter(df[x], df[y])
            plt.title(f"{x} vs. {y}")
            plt.savefig(f"reference_figures/visualizers_test/test_{x}_vs_{y}_scatter.png")
            plt.clf()

    if args.ridgeplots:
        print(df.head())
        assert "ID" in df, "There's only one group in this DF!"
        plots = [g for g in args.ridgeplots.split(",")]
        mapping = {1:"#fde725FF", 
                   2:"#90d743FF", 
                   3:"#35b779FF", 
                   4:"#21918cFF", 
                   5:"#31688eFF", 
                   6:"#443983FF", 
                   7:"#440154FF"}
        # df["colors"]=df["dpg"].map(mapping)
        # group_df = df.groupby(["dpg"])
        for plot in plots:
            fig, ax = joypy.joyplot(df, by = "ID", column = plot, fade = True, color = "blue")
            #fig, axes = joypy.joyplot(df, by="Team", column="Minute", colormap = cmap)
            plt.title(f"Ridgeplot of {plot}")
            plt.savefig(f"reference_figures/visualizers_test/test_{plot}_ridgeplot.png")
            plt.clf()

        # plt.show()


    return True

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_data', type=str, required=True, help='Path to image data table')
    parser.add_argument('--histograms', type=str, help='Sets of histograms to build and save')
    parser.add_argument('--scatterplots', type=str, help='Sets of scatterplots to build and save')
    parser.add_argument('--ridgeplots', type=str, help='Sets of ridgeplots to build')
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')



#    #TODO: FIGURE OUT How to store the results of O(n^2) distances to a file specific to an image being analyzed
# def histogram():
#     if args.histogram:
#         for param in args.histogram.split("%"):
#         # plt.hist(clump_info_dict.values(), bins=list(range(0, 1000, 50)))
#         # plt.title("Clump Sizes")
#         # plt.show()
#         #print(clump_info_dict)
#         #assert 'area' in table.keys(), "Area is not listed in this table!"

#             plt.hist(table[param])#, bins=list(range(0, 2000, 120)))
#             plt.title(f"{param}")
#             plt.figure(figsize = (8,6))
#             if args.save_to:
#                 plt.savefig(f"reference_figures/{args.save_to}_area_histogram.png")
#             #plt.show()
#             plt.clf()
        
        
# def scatterplot():
#     if args.scatter_plot:
#         for param in args.scatter_plot.split("%"):
#             #assert 'axis_major_length' in table.keys(), "Major Axis Length is not listed in this table!"
#             field1, field2 = param.split("^")
#             plt.figure(figsize = (8,6))
#             plt.scatter(table[field1], table[field2])
#             # plt.xlim(0, 2000)
#             # plt.ylim(0,   60)
#             plt.title(f"{field1} vs. {field2}")
            
#             if args.save_to:
#                 plt.savefig(f"reference_figures/{args.save_to}_{field1}_vs_{field2}_scatter.png")
#             else:
#                 plt.show()
#             #plt.show()
#             plt.clf()
            
#             heatmap, xedges, yedges = numpy.histogram2d(table[field1], table[field2])#, bins = [list(range(0, 2000, 50)), list(range(0, 1000, 50))])
#             extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

#             plt.clf()
#             plt.imshow(heatmap.T, extent=extent, origin='lower')
#             if args.save_to:
#                 plt.savefig(f"reference_figures/{args.save_to}_{field1}_vs_{field2}_heatmap.png")
#             else:
#                 plt.show()
#             #plt.show()
#             plt.clf()
    
#     if args.distances:
#         #print(table)
#         #print(table['centroid-0'])
#         #print(table['centroid-1'])

#         points = [(x,y) for x in table['centroid-0'] for y in table['centroid-1']]
        

#     return True