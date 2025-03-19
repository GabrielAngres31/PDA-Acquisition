import argparse
import os
import typing as tp

import PIL.Image
import torch

import src.data

import tqdm
from timeit import default_timer as timer


def main(args:argparse.Namespace) -> bool:
    model = torch.load(args.model).eval()
    x     = src.data.load_inputimage(args.input)
    y     = run_patchwise_inference(model, x, args.overlap)
    outf  = os.path.join(args.outputdir, os.path.basename(args.input)+f'{args.outputname}.output.png')
    # print("DIR_OUTPUT IS:")
    # print(os.path.dirname(outf))
    
    os.makedirs(os.path.dirname(outf), exist_ok=True)
    src.data.save_image(outf, y[0,0].numpy())
    return True


def run_patchwise_inference(model:torch.nn.Module, x:torch.Tensor, overlap:int) -> torch.Tensor:
    input_patches  = src.data.slice_into_patches_with_overlap(x, 256, overlap)
    output_patches = []
    if args.progress: iterator = tqdm.tqdm(input_patches)
    else: iterator = input_patches
    for x_patch in iterator:
        with torch.no_grad():
            y_patch = model(x_patch[None])
        output_patches.append(y_patch)
    result = src.data.stitch_overlapping_patches(output_patches, x.shape, overlap)
    result = result.sigmoid()
    return result


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to saved model')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--close_on', type=int, help='Maximum hole size to remove')
    parser.add_argument('--open_on', type=int, help='Minimum clump size to keep')
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
        help = "Whether to show a progress bar. May disable for batch jobs."
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
