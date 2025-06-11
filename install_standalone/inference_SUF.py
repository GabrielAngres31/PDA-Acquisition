import argparse
import os
import typing as tp

import PIL.Image
import torch

import src.data

import tqdm
from timeit import default_timer as timer
import numpy as np


def main(args:argparse.Namespace) -> bool:
    # print(args.weights_only)
    model = torch.load(args.model, weights_only=False).eval()
    x     = src.data.load_inputimage(args.input)
    y     = run_patchwise_inference(model, x, args.patchsize, args.overlap, args.skip_empty)
    outf  = os.path.join(args.outputdir, os.path.basename(args.input)+f'{args.outputname}.output.png')
    # print(outf)
    # print("DIR_OUTPUT IS:")
    # print(os.path.dirname(outf))
    
    os.makedirs(os.path.dirname(outf), exist_ok=True)
    src.data.save_image(outf, y[0,0].numpy())
    return True


def run_patchwise_inference(model:torch.nn.Module, x:torch.Tensor, patchsize:int, overlap:int, skip_empty:bool) -> torch.Tensor:
    # Slice input image into parseable NxN patches for the model
    input_patches  = src.data.slice_into_patches_with_overlap(x, patchsize=patchsize, overlap=overlap)
    # Create list for output patches.
    output_patches = []
    # If the progress flag is activated, show a tqdm progress bar.
    # Otherwise, infer silently.
    if args.progress: iterator = tqdm.tqdm(input_patches) 
    else: iterator = input_patches
    # Pregenerate a list of pixels along the border to check for "relevance". If none of these pixels are sufficiently bright, the inference loop skips that frame and moves to the next one.
    indices_check = [n for x in range(0, patchsize, 16) for n in [(x,0), (patchsize-1, x), (patchsize-1-x, patchsize-1), (0, patchsize-1-x)]]

    # Inference loop start:
    for x_patch in iterator:
        # If skipping is turned on, do the following:
        if args.skip_empty:
            # Check if any of the pixels in the pregenerated indices list are "bright" enough to be relevant.
            # This value is hand-tuned - minimum and maximum ranges for tensor values are not normalized to [0,255] so I picked what appeared to work in the general case.
            # RESULTS MAY VARY!
            check = not any([x_patch[0][a,b]>0.25 for (a,b) in indices_check])
            if check:
                # Write -10 to patch, which upon applying thr sigmoid function (1/[1+exp(-x)]) will evaluate to 0.0003 (<< 1/256 = ~0.004)
                # Which will be converted to a #000000 or BLACK pixel on subsequent normalization to [0, 255] monochrome.
                y_patch = torch.full((1, 1, 256, 256), 0)
                output_patches.append(y_patch)
                # Move to the next patch.
                continue
        # If skipping is not enabled, or the patch has been found to be "interesting", the patch is evaluated by the model.
        with torch.no_grad():
            y_patch = model(x_patch[None])
        output_patches.append(y_patch)
    # Stitch patches together into a single image
    result = src.data.stitch_overlapping_patches(output_patches, x.shape, overlap)

    result = result.sigmoid()
    return result

def check_indices(wsize, step):
    i = 0
    while step*i < wsize:
        yield [(i*step,0), (wsize, i*step), (wsize-i*step,wsize), (0,wsize-i*step)]
        i+=1
        


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to saved model')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--close_on', type=int, help='Maximum hole size to remove')
    parser.add_argument('--open_on', type=int, help='Minimum clump size to keep')
    parser.add_argument('--patchsize', type=int, default=256, help='Patch size, default 256')
    parser.add_argument('--overlap', type=int, help='Vertical overlap between patches')
    parser.add_argument(
        '--outputdir', 
        type    = str, 
        default = './inference/', 
        help    = 'Path to outputfolder'
    )
    parser.add_argument(
        '--outputname', 
        type    = str, 
        default = '', 
        help    = 'Optional suffix before file type'
    )
    parser.add_argument(
        "--progress",
        type=str,
        help    = "Whether to show a progress bar. May disable for batch jobs."
    )
    parser.add_argument(
        "--weights_only",
        default = True
    )
    parser.add_argument(
        "--skip_empty",
        default = False,
        help = "Whether the sliding window skips frames that are empty. This is determined by checking the average of the top, bottom, left, and right border pixels. If any of these has an average of more than 1 out of 255, the frame is included."
    )
    return parser



if __name__ == '__main__':
    t0 = timer()
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        t1 = timer()
        elapsed = t1-t0
        print(f'Done in {elapsed} seconds')
