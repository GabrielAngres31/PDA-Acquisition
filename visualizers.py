# visualizers

import 


   #TODO: FIGURE OUT How to store the results of O(n^2) distances to a file specific to an image being analyzed
def histogram():
    if args.histogram:
        for param in args.histogram.split("%"):
        # plt.hist(clump_info_dict.values(), bins=list(range(0, 1000, 50)))
        # plt.title("Clump Sizes")
        # plt.show()
        #print(clump_info_dict)
        #assert 'area' in table.keys(), "Area is not listed in this table!"

            plt.hist(table[param])#, bins=list(range(0, 2000, 120)))
            plt.title(f"{param}")
            plt.figure(figsize = (8,6))
            if args.save_to:
                plt.savefig(f"reference_figures/{args.save_to}_area_histogram.png")
            #plt.show()
            plt.clf()
        
        
def scatterplot():
    if args.scatter_plot:
        for param in args.scatter_plot.split("%"):
            #assert 'axis_major_length' in table.keys(), "Major Axis Length is not listed in this table!"
            field1, field2 = param.split("^")
            plt.figure(figsize = (8,6))
            plt.scatter(table[field1], table[field2])
            # plt.xlim(0, 2000)
            # plt.ylim(0,   60)
            plt.title(f"{field1} vs. {field2}")
            
            if args.save_to:
                plt.savefig(f"reference_figures/{args.save_to}_{field1}_vs_{field2}_scatter.png")
            else:
                plt.show()
            #plt.show()
            plt.clf()
            
            heatmap, xedges, yedges = numpy.histogram2d(table[field1], table[field2])#, bins = [list(range(0, 2000, 50)), list(range(0, 1000, 50))])
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            plt.clf()
            plt.imshow(heatmap.T, extent=extent, origin='lower')
            if args.save_to:
                plt.savefig(f"reference_figures/{args.save_to}_{field1}_vs_{field2}_heatmap.png")
            else:
                plt.show()
            #plt.show()
            plt.clf()
    
    if args.distances:
        #print(table)
        #print(table['centroid-0'])
        #print(table['centroid-1'])

        points = [(x,y) for x in table['centroid-0'] for y in table['centroid-1']]
        

    return True