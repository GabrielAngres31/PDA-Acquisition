import argparse
import typing as tp

import torchvision

import src.training_utils


def main(args: argparse.Namespace) -> bool:
    """Training entry point"""
    trainfiles = args.trainingfolder
    validationfiles = None

    if args.validationfolder is not None:
        validationfiles = (args.validationfolder,)

    model = torchvision.models.mobilenet_v3_large(weights="DEFAULT")
    assert isinstance(trainfiles, tp.Iterable), "WOOPS"
    model = src.training_utils.run_training_mbn(
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
        "--trainingfolder",
        type=str,
        required=True,
        help="""Path to folder containing class-labeled images to be used for training""",
    )
    parser.add_argument(
        "--validationfolder",
        type=str,
        required=False,
        help="""Path to folder containing class-labeled images to be used for validation""",
    )
    parser.add_argument(
        "--checkpointdir",
        type=str,
        default="./checkpoints/mbn/",
        help="Where to store trained models",
    )
    parser.add_argument(
        "--outputcsv",
        type=str,
        required=True,
        help="Name to store csv of losses and epochs for easy viewing",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for AdamW optimizer"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=16,
        help="Number of samples in a batch per training step",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help="Extra training weight on the positive class",
    )
    return parser


if __name__ == "__main__":
    args = get_argparser().parse_args()
    ok = main(args)
    if ok:
        print("Done")
