# import training_utils
import src.data
import torch
# from training_utils import validate_one_epoch as epval
import argparse
import numpy as np
import typing as tp

# import modelsets/model from filepath

# Get expanded validation set from csv
# Generate dataloader
# calculate validation loss and output to console

def epval(module:torch.nn.Module, loader:tp.Iterable) -> float:
    losses = []
    for i,[x,t] in enumerate(loader):
        with torch.no_grad():
            y    = module(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y, t)
        losses.append(loss.item())
    return {"mean":np.mean(losses), "losses":losses}

def main(args:argparse.Namespace) -> bool:
    module = torch.load(args.model, weights_only=False).eval()

    mean_vloss_list = []
    all_vloss_list  = []

    # module.train()

    validation_filepairs = src.data.load_splitfile(args.validationsplit)
    validation_filepairs = src.data.cache_file_pairs(
        validation_filepairs, args.cachedir, args.patchsize, args.overlap, clear=False
    )


    vdataset = src.data.Dataset(validation_filepairs)
    vloader  = src.data.create_dataloader(vdataset, args.batchsize, shuffle=False)

    vloss_info = epval(module, vloader)
    vloss, vloss_all = vloss_info["mean"], vloss_info["losses"]
    print(vloss)
    # print(vloss_all)
    

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', 
        type     = str, 
        required = True,
        help     = '''Path to model to validate.'''
    )
    parser.add_argument(
        '--validationsplit', 
        type     = str, 
        required = False,
        help     = '''Path to .csv file containing pairs of input images and 
        annotations to be used for validation'''
    )
    parser.add_argument(
        '--patchsize',
        type    = int,
        default = 128,
        help    = 'Size of input patches in pixels'
    )
    parser.add_argument(
        '--cachedir', 
        type    = str, 
        default = './cache_standalone_val/', 
        help    = 'Where to store image patches',
    )
    parser.add_argument(
        '--overlap',
        type    = int,
        default = 16,
        help    = 'How much overlap between patches',
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

    # start_time = time.time()
    ok   = main(args)
    # end_time = time.time()

    # execution_time = end_time - start_time
    # print("Execution time:", execution_time)
    
    if ok:
        pass
        # print('Done')

