import argparse
import os
import time
import typing as tp

import numpy as np
import torch
import torchvision
import PIL.Image

import src.data


import sys

def main(args:argparse.Namespace) -> bool:
    grnd = src.data.load_image(args.ground_truth,  "RGB")
    pred = src.data.load_image(args.model_predict, "RGB")
    assert grnd.shape == pred.shape
    diff = torch.sub(grnd, pred)
    pos_mask = torch.where(diff > 0,  diff, 0)
    neg_mask = torch.where(diff < 0, -diff, 0)

    transform = torchvision.transforms.ToPILImage()
    img_neg = transform(neg_mask)
    img_pos = transform(pos_mask)
    #img.convert("RGB")
    pixels_neg = img_neg.load()
    pixels_pos = img_pos.load()
    fp_col = args.false_pos_col
    tp_col = args.true_pos_col
    #img_neg.show()
    #img_pos.show()
    hex_norm = lambda c, l: (c[0]*l//255, c[1]*l//255, c[2]*l//255)
    
    for c in range(img_neg.size[0]):
        for r in range(img_neg.size[1]):
            #print(pixels[c,r][0])
            pixels_neg[c,r]=(hex_norm(fp_col, pixels_neg[c,r][0]))
    
    for c in range(img_pos.size[0]):
        for r in range(img_pos.size[1]):
            #print(pixels[c,r][0])
            pixels_pos[c,r]=(hex_norm(tp_col, pixels_pos[c,r][0]))

    img_both = PIL.Image.new('RGB', (img_neg.size[0], img_neg.size[1]), "black")
    pixels_both = img_both.load()
    for c in range(img_neg.size[0]):
        for r in range(img_neg.size[1]):
            assert pixels_pos[c,r] == (0,0,0) or pixels_neg[c,r] == (0,0,0)
            #print(pixels_pos[c,r])
            if pixels_pos[c,r] == (0,0,0):
                pixels_both[c,r] = pixels_neg[c,r]
            else:
                pixels_both[c,r] = pixels_pos[c,r]

    if args.show: 
        img_both.show()


    if args.save:
        img_both.save(args.save)

    return True



def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ground_truth',
        type = str,
        required = True,
        help = 'Path to image containing manual annotations.'
    )
    parser.add_argument(
        '--model_predict',
        type = str,
        required = True,
        help = 'Path to image containing prediction on base image.'
    )
    parser.add_argument(
        '--show',
        type = int,
        required = False,
        default = 0,
        help = 'Enable to show image difference visualizations'
    )
    parser.add_argument(
        '--save',
        type = str,
        required = False,
        default = '',
        help = 'Enable to save resulting image to given path.'
    )
    parser.add_argument(
        '--false_pos_col',
        type = str,
        default = [255,165,0],
        help = 'Color to highlight false positives (umbrella, no rain)'
    )
    parser.add_argument(
        '--true_pos_col',
        type = str,
        default = [0,90,255],
        help = 'Color to highlight true positives (umbrella, rain)'
    )
    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')

        