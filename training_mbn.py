import argparse
import os
import time
import typing as tp

import numpy as np
import torch
import torchvision

import src.data
import src.training_utils_clumpdiff
import src.unet
import mbn


def main(args:argparse.Namespace) -> bool:
    '''Training entry point'''
    
    def get_subfolders(path):
        folders = []
        for entry in os.scandir(path):
            if entry.is_dir():
                folders.append(entry.name)
        return folders
    
    trainfiles = src.data.load_labeled_images(get_subfolders(args.training_folders_parent))
    validationfiles = None
    if args.validation_folders_parent is not None:
        validationfiles - src.data.load_labeled_images(get_subfolders(args.validation_folders_parent))
    
    # THE STEPS ABOVE SHOULD GENERATE SOME DATA INSTEAD

    model = mbn.MobileNetV3LC()

    model = src.training_utils_clumpdiff.run_training_mbn(
        model, 
        trainfiles, 
        args.epochs, 
        args.lr,
        args.batchsize,
        args.pos_weight,
        args.checkpointdir, 
        args.outputcsv,
        validationfiles,
    )
    return True



def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--training_folders_parent',
        type    = str,
        help = 'Parent folder for labeled folders containing training data.'
    )
    parser.add_argument(
        '--validation_folders_parent',
        type    = str,
        help = 'Parent folder for labeled folders containing validation data.'
    )
    parser.add_argument(
        '--checkpointdir',
        type    = str,
        default = './checkpoints/',
        help    = 'Where to store trained models',
    )
    parser.add_argument(
        '--outputcsv',
        type    = str,
        default = '',
        help    = 'Name to store csv of losses and epochs for easy viewing',
    )
    parser.add_argument(
        '--epochs',
        type    = int,
        default = 30,
        help    = 'Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type    = float,
        default = 1e-3,
        help    = 'Learning rate for AdamW optimizer'
    )
    parser.add_argument(
        '--batchsize',
        type    = int,
        default = 8,
        help    = 'Number of samples in a batch per training step'
    )
    parser.add_argument(
        '--pos_weight',
        type    = float,
        default = 5.0,
        help    = 'Extra training weight on the positive class'
    )
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')
