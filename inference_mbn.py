import argparse
import os
import typing as tp

import PIL.Image
import torch

import src.data

import tqdm
from timeit import default_timer as timer


def main(args:argparse.Namespace) -> bool:
    classer = lambda f: "Single" if f else "Clustered"
    assert args.input_image or args.input_folder, "You haven't provided any input images or folders!"
    p = args.input_image if args.input_image else args.input_folder
    model = torch.load(args.model, weights_only=False).eval()
    if args.input_image:
        x     = src.data.load_inputimage(args.input_image)
        y = run_inference(model, x)
        y = torch.softmax(y, dim=1, dtype=float)
        y = torch.argmax(y, dim=1)
        # print(p, ": ", classer(y[0]))
        return(classer(y[0]))
    elif args.input_folder:
        dir_list = os.listdir(args.input_folder)
        x     = torch.stack([src.data.load_inputimage(os.path.join(args.input_folder, f)) for f in dir_list])
        y = run_inference(model, x)
        y = torch.softmax(y, dim=1, dtype=float)
        y = torch.argmax(y, dim=1)
        # [print(f"{dir_list[i]}: \t{classer(y[i])}") for i, t in enumerate(y)]
        with open("C:/Users/Gabriel/Documents/Github/PDA-Acquisition/SCD_training_data/mbn_training/guess_outfile.txt", "w") as t:
            for l in [(f"{dir_list[i]}: \t{classer(y[i])}") for i, t in enumerate(y)]:
                t.write(f"{l}\n")
    return True


def run_inference(model:torch.nn.Module, x:torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if x.dim() == 3:
            result = model(x[None])
        elif x.dim() == 4:
            result = model(x)
    return result


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to saved model')
    parser.add_argument('--input_image',  type=str, required=False, help='Path to input image')
    parser.add_argument('--input_folder', type=str, required=False, help='Path to folder with input images')
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
    return parser



if __name__ == '__main__':
    t0 = timer()
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print(ok)
        # t1 = timer()
        # elapsed = t1-t0
        # print(f'Done in {elapsed} seconds')
