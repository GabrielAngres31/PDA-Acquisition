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
    # if not args.xmax:

    df = pd.read_csv(args.source_data)
    #print(args.source_data)
    #print(df)

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
            plt.savefig(f"reference_figures/visualizers_test/{args.save_as}_{measure}_histogram.png")
            plt.clf()

    def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y)

        # now determine nice limits by hand:
        binwidth = 0.25
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=bins)
        ax_histy.hist(y, bins=bins, orientation='horizontal')


    if args.scatterplots:
        assert not ("ID" in df), "Too many groups!"
        num = len(df)
        plots=args.scatterplots.split("|")
        cluster_mapping = {1:"#fde725FF",
                           2:"#21918cFF",
                           3:"#440154FF"}
        for plot in plots:
            #print(plot)
            x,y = plot.split(",")
            """ This section is for normal scatter plotting"""
            plt.figure(figsize = (8,6))
            if not ("type" in df):
                plt.scatter(df[x], df[y])
            else:
                plt.scatter(df[x], df[y], c=df["type"].map({3:"#fde725FF", 2:"#21918cFF", 1:"#440154FF"}))
            plt.axis(xmin=0, ymin=0, xmax=args.xmax, ymax=args.ymax)
            plt.text(0.5, 0.5, f'N = {num}', ha='right')
            plt.title(f"{x} vs. {y} - {args.save_as}")
            plt.savefig(f"reference_figures/visualizers_test/{args.save_as}_{x}_vs_{y}_scatter.png")
            plt.clf()
            """This section is for having histograms on the sides"""\
            """Code from https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html"""
            # fig = plt.figure(figsize=(6, 6))
            # # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
            # # the size of the marginal Axes and the main Axes in both directions.
            # # Also adjust the subplot parameters for a square plot.
            # gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
            #                     left=0.1, right=0.9, bottom=0.1, top=0.9,
            #                     wspace=0.05, hspace=0.05)
            # # Create the Axes.
            # ax = fig.add_subplot(gs[1, 0])
            # ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
            # ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
            # # Draw the scatter plot and marginals.
            # scatter_hist(x, y, ax, ax_histx, ax_histy)
            # plt.savefig(f"reference_figures/visualizers_test/{args.save_as}_{x}_vs_{y}_scatter_w-hist.png")
            plt.clf()

    if args.density_heatmaps:
        assert not ("ID" in df), "Too many groups!"
        num = len(df)
        plots=args.density_heatmaps.split("|")

        for plot in plots:
            #print(plot)
            x,y = plot.split(",")
            """ This section is for normal scatter plotting"""
            plt.figure(figsize = (8,6))
            if not ("type" in df):
                plt.hist2d(df[x], df[y], cmap="viridis", bins=40, range=[[0, args.xmax],[0, args.ymax]])
                # plt.hexbin(df[x], df[y], cmap="viridis", gridsize=40, bins='log', extent=(0, args.xmax, 0, args.ymax))
            # else:
            #     plt.hist2d(df[x], df[y], c=df["type"].map({3:"#fde725FF", 2:"#21918cFF", 1:"#440154FF"}))
            plt.axis(xmin=0, ymin=0, xmax=args.xmax, ymax=args.ymax)
            plt.text(0.5, 0.5, f'N = {num}', ha='right')
            plt.title(f"{x} vs. {y} - {args.save_as}")
            plt.savefig(f"reference_figures/visualizers_test/{args.save_as}_{x}_vs_{y}_hex_density_heatmap.png")
            plt.clf()

    if args.ridgeplots:
        
        #print(df.head())
        assert "ID" in df, "There's only one group in this DF!"
        plots = [g for g in args.ridgeplots.split(",")]
        dpg_mapping = {1:"#fde725FF", 
                   2:"#90d743FF", 
                   3:"#35b779FF", 
                   4:"#21918cFF", 
                   5:"#31688eFF", 
                   6:"#443983FF", 
                   7:"#440154FF"}
        AZD_mapping = {
            1:"#fde725FF",
            2:"#35b779FF",
            3:"#31688eFF",
            4:"#440154FF"
        }

        sorted_count_by_id = df.groupby("ID").size().sort_values(ascending=True)
        sortcount_to_dict = dict(zip([k for k in sorted_count_by_id.keys()], 
                                     [v for v in sorted_count_by_id.values]))
        



        df["num"] = df.apply(lambda row: sortcount_to_dict[row["ID"]], axis=1)
        #print(df.head())
        grouped_by_ID = df.groupby("ID", sort=False)
        # NOTE: this assumes that all items in a group have the same dpg
        dpg_by_ID    = [ list(group['dpg'])[0]  for _, group in grouped_by_ID]
        colors_by_ID = [ dpg_mapping[dpg] for dpg in dpg_by_ID ]
        
        for plot in plots:
            fig, ax = joypy.joyplot(df.groupby("ID", sort=False), by = "ID", column = plot, fade = True, color = colors_by_ID)
            plt.title(f"Ridgeplot of {plot}")
            plt.savefig(f"reference_figures/visualizers_test/{args.save_as}_{plot}_dpg_ridgeplot.png", bbox_inches = "tight")
            plt.clf()
        df = df.sort_values("num", ascending=True)
        print(df.head())
        for plot in plots:
            # fig, ax = joypy.joyplot(df, by="num", column = plot, fade = True)
            fig, ax = fig, ax = joypy.joyplot(df.groupby("ID", sort=False), column = plot, fade = True, color = colors_by_ID)
            plt.title(f"Ridgeplot of {plot} - {args.save_as}")
            plt.savefig(f"reference_figures/visualizers_test/{args.save_as}_{plot}_num_ridgeplot.png", bbox_inches = "tight")
            plt.clf()
        # print(str(grouped_by_ID.size()))
        # print(str(grouped_by_ID.size().sort_values(ascending=True)))

        colors_by_norm = []

        # plt.show()


    return True

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_data', type=str, required=True, help='Path to image data table')
    parser.add_argument('--histograms', type=str, help='Sets of histograms to build and save')
    parser.add_argument('--scatterplots', type=str, help='Sets of scatterplots to build and save')
    parser.add_argument('--ridgeplots', type=str, help='Sets of ridgeplots to build and save')
    parser.add_argument('--density_heatmaps', type=str, help='Sets of density heatmaps to build and save')
    parser.add_argument('--save_as', type=str, help='File name to save to')
    parser.add_argument('--xmax', type=int, help='Length of X axis for scatterplots')
    parser.add_argument('--ymax', type=int, help='Length of Y axis for scatterplots')

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