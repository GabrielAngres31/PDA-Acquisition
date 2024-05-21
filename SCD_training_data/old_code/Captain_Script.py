import argparse
import os

DIR_BASE = "SCD_training_data\\source_images\\BASE"
DIR_MASK = "SCD_training_data\\source_images\\ANNOTATION"


parser = argparse.ArgumentParser()

parser.add_argument("--predict", action = "store", help = "select a file to predict stoma locations within the image")
parser.add_argument("--evaluate", help = "provide user-viewable display of training accuracy")
parser.add_argument("--train", help = "train machine learning algorithm on split folders")
parser.add_argument("--split", help = "split existing paired base/mask data into split folders on the hard drive")
parser.add_argument("--scrape", help = "generate paired base/mask data from the images within a folder")

args = parser.parse_args()




if args.scrape:
    pass


if args.predict:
    print(args.predict)


