import argparse
import os
import time
import typing as tp

import numpy as np
import torch
import torchvision

import src.data
import src.training_utils
import src.unet


def main(args:argparse.Namespace) -> bool:
    '''Training entry point'''
    print(args.trainingfolder)
    trainfiles = src.data.load_splitfile(args.trainingfolder)
    trainfiles = src.data.cache_file_pairs(
        trainfiles, args.cachedir, args.patchsize, args.overlap
    )
    validationfiles = None
    if args.validationsplit is not None:
        validationfiles = src.data.load_splitfile(args.validationfolder)
        validationfiles = src.data.cache_file_pairs(
            validationfiles, args.cachedir, args.patchsize, args.overlap, clear=False
        )
    


    model = src.unet.UNet() # TODO: CHANGE THE MODEL TO MOBILENET
    model = src.training_utils.run_training_mbn(
        model, 
        trainfiles, 
        args.epochs, 
        args.lr,
        args.batchsize,
        args.pos_weight,
        args.checkpointdir, 
        #args.model_ID,
        args.outputcsv,
        validationfiles,
    )
    return True



def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trainingfolder', 
        type     = str, 
        required = True,
        help     = '''Path to folder containing class-labeled images to be used for training'''
    )
    parser.add_argument(
        '--validationfolder', 
        type     = str, 
        required = False,
        help     = '''Path to folder containing class-labeled images to be used for validation'''
    )
    parser.add_argument(
        '--patchsize',
        type    = int,
        default = 256,
        help    = 'Size of input patches in pixels'
    )
    parser.add_argument(
        '--cachedir', 
        type    = str, 
        default = './cache/', 
        help    = 'Where to store image patches',
    )
    parser.add_argument(
        '--checkpointdir',
        type    = str,
        default = './checkpoints/',
        help    = 'Where to store trained models',
    )
    # parser.add_argument(
    #     'model_ID',
    #     type = str,
    #     default = "",
    #     help = 'Identifier to put before the date in the model name',
    # )

    parser.add_argument(
        '--overlap',
        type    = int,
        default = 32,
        help    = 'How much overlap between patches',
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
