import argparse
import os
import typing as tp

import PIL.Image
import torch

import src.data

import tqdm


def main(args:argparse.Namespace) -> bool:
    model = torch.load(args.model).eval()
    x     = src.data.load_inputimage(args.input)
    y     = run_patchwise_inference(model, x)
    outf  = os.path.join(args.outputdir, os.path.basename(args.input)+'.output.png')
    os.makedirs(os.path.dirname(outf), exist_ok=True)
    src.data.save_image(outf, y[0,0].numpy())
    return True


def run_patchwise_inference(model:torch.nn.Module, x:torch.Tensor) -> torch.Tensor:
    #FIXME: hardcoded patchsize
    input_patches  = src.data.slice_into_patches_with_overlap(x, 256, 32)
    output_patches = []
    for x_patch in tqdm.tqdm(input_patches):
        with torch.no_grad():
            y_patch = model(x_patch[None])
        output_patches.append(y_patch)
    result = src.data.stitch_overlapping_patches(output_patches, x.shape, 32)
    result = result.sigmoid()
    return result


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to saved model')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument(
        '--outputdir', 
        type    = str, 
        default = './inference/', 
        help    = 'Path to outputfolder'
    )
    return parser



if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')
