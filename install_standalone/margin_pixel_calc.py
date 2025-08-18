# margin_pixel_calc

from PIL import Image
import numpy
import argparse


def main(args:argparse.Namespace) -> bool:
    
    image_in = Image.open((args.input_path))
    pixelheap = image_in.load()
    
    width, height = image_in.size

    marginpixels = []

    for x in range(width):
        for y in range(height):
            pixel = pixelheap[x,y]
            # print(pixel)
            if pixel:
                plist = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                for p in plist:
                    try:
                        if pixelheap[p[0], p[1]] == 0:
                            marginpixels.append(p)
                    except IndexError:
                        continue
    unique_marginpixels = list(set(marginpixels))
    # print("Number of margin pixels:", len(unique_marginpixels))
    # print(len(marginpixels))
    # print(len(unique_marginpixels))
                # print(pixel)
                # print(x, ",", y)

    blurpixels = []
    for u in unique_marginpixels:
        ux = u[0]
        uy = u[1]
        # [blurpixels.append(r) for r in [(ux, uy), (ux, uy - 1), (ux, uy + 1), (ux - 1, uy), (ux + 1, uy), (ux - 2, uy), (ux + 2, uy), (ux, uy - 2), (ux, uy + 2), (ux - 1, uy - 1), (ux - 1, uy + 1), (ux + 1, uy - 1), (ux + 1, uy + 1)]]
        [blurpixels.append(r) for r in [(ux, uy), (ux, uy - 1), (ux, uy + 1), (ux - 1, uy), (ux + 1, uy), (ux - 1, uy - 1), (ux - 1, uy + 1), (ux + 1, uy - 1), (ux + 1, uy + 1)]]
    
    # print(len(blurpixels))
    unique_blurpixels = list(set(blurpixels))
    # print("Number of blur pixels:", len(unique_blurpixels))
    print("FINALNUM =", len(unique_blurpixels)-len(unique_marginpixels))


    if args.compare_path:
        d = 0
        comp_img = Image.open(args.compare_path)
        assert comp_img.size == image_in.size
        pix_comp_img = comp_img.load()

        for i in range(width):
            for j in range(height):
                try: 
                    if pixelheap[i,j] != pix_comp_img[i,j]:
                        d += 1
                except IndexError:
                    continue
    # print("Discrepant pixels:", d)

    # print(f"Fold Error: {d/len(unique_blurpixels)}")



def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type = str,
        required = True,
        help = 'Image to profile margin pixels for.'
    )
    parser.add_argument(
        '--compare_path',
        type = str,
        # required = True,
        help = 'Image to compare against.'
    )
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')
