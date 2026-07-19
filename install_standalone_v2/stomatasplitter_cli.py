#!/usr/bin/env python3
"""
stomatasplitter.py

This script parses stomata-centered sections from whole images in order to train the
clump-detecting network. It automatically sorts the stomata into a training/validation split.
Requires annotated images as an input (in grabfile.csv).
"""

import argparse
import glob
import os

import pandas as pd
from PIL import Image


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract stomata image crops for dataset generation."
    )
    parser.add_argument(
        "--grab_list",
        type=str,
        default="grabfile.csv",
        help="Path to the main grab CSV file (default: grabfile.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_folder_clumpcatcher",
        help="Directory to save extracted crops (default: training_folder_clumpcatcher)",
    )
    parser.add_argument(
        "--additional_dir",
        type=str,
        default="additional_clump_images",
        help="Directory containing augmented data (default: additional_clump_images)",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=72,
        help="Total width/height of the square crop (default: 72)",
    )
    parser.add_argument(
        "--split_ratio",
        type=int,
        default=5,
        help="Every Nth image goes to validation to achieve an N-1/1 split (default: 5)",
    )
    parser.add_argument(
        "--keep_existing",
        action="store_true",
        help="Set this flag to prevent clearing the output folder before running",
    )
    return parser


def clear_directory(directory: str) -> None:
    """Removes all files inside the specified folder safely."""
    if os.path.exists(directory):
        print(f"Clearing existing files in: {directory}")
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")


def main(args: argparse.Namespace) -> bool:
    # 1. Parameter Initialization & Path Management
    half_crop = args.crop_size // 2
    output_dir = args.output_dir

    # Starting seed counters to ensure deterministic train/val sequencing
    # A value of (split_ratio - 1) makes the first item land on index % split_ratio != 0 (train)
    start_counter = args.split_ratio - 1
    s = start_counter  # Single counter
    c = start_counter  # Cluster counter
    q = start_counter  # Additional counter

    # Clear directories unless explicitly told not to
    if not args.keep_existing:
        clear_directory(output_dir)

    # 2. Process Primary Grab List CSV
    if not os.path.exists(args.grab_list):
        print(f"Error: Main grab file '{args.grab_list}' not found.")
        return False

    grab_df = pd.read_csv(args.grab_list)
    print("Processing primary images dataset...")
    print(grab_df.head())

    savedict = {"Hit": "single", "Cl. Hit": "cluster"}
    numdict = {"Hit": 0, "Cl. Hit": 1}

    for _, row_vals in grab_df.iterrows():
        base = Image.open(row_vals["base"])
        clumps = pd.read_csv(row_vals["clumps"])

        for idx, clm_vals in clumps.iterrows():
            y0, x0, y1, x1 = (
                clm_vals["bbox-0"],
                clm_vals["bbox-1"],
                clm_vals["bbox-2"],
                clm_vals["bbox-3"],
            )

            # Center coordinates calculation
            xc, yc = int((x0 + x1) // 2), int((y0 + y1) // 2)
            xl, xr, yu, yd = (
                xc - half_crop,
                xc + half_crop,
                yc - half_crop,
                yc + half_crop,
            )

            # Determine split (train vs val) dynamically
            is_cluster = numdict[clm_vals["Notes"]]
            counter_val = c if is_cluster else s
            split_tag = "train" if counter_val % args.split_ratio else "val"
            class_tag = savedict[clm_vals["Notes"]]

            # Structural subfolder generation
            target_subfolder = os.path.join(output_dir, split_tag, class_tag)
            os.makedirs(target_subfolder, exist_ok=True)

            # Crop and save execution
            clump_crop = base.crop((xl, yu, xr, yd))
            base_filename = os.path.splitext(os.path.basename(row_vals["base"]))[0]
            save_name = f"{base_filename}_{idx}_{xc:04d}x_{yc:04d}y.png"
            clump_crop.save(os.path.join(target_subfolder, save_name))

            # Multi-counter increment tracking
            s += is_cluster
            c += 1 - is_cluster

    # 3. Process Secondary/Augmented Dataset
    additional_csv_pattern = os.path.join(args.additional_dir, "clumps", "*.csv")
    additional_files = glob.glob(additional_csv_pattern)

    if additional_files:
        print(f"Processing augmented data from {args.additional_dir}...")
        for csv_path in additional_files:
            clfl = pd.read_csv(csv_path)
            base_img_name = os.path.splitext(os.path.basename(csv_path))[0]
            matching_img_path = os.path.join(
                args.additional_dir, "base", f"{base_img_name}.tif"
            )

            if not os.path.exists(matching_img_path):
                print(f"Warning: Expected image variant missing: {matching_img_path}")
                continue

            img = Image.open(matching_img_path)

            for idx, p_vals in clfl.iterrows():
                y0, x0, y1, x1 = (
                    p_vals["bbox-0"],
                    p_vals["bbox-1"],
                    p_vals["bbox-2"],
                    p_vals["bbox-3"],
                )
                xc, yc = int((x0 + x1) // 2), int((y0 + y1) // 2)
                xl, xr, yu, yd = (
                    xc - half_crop,
                    xc + half_crop,
                    yc - half_crop,
                    yc + half_crop,
                )

                split_tag = "train" if q % args.split_ratio else "val"
                target_subfolder = os.path.join(output_dir, split_tag, "cluster")
                os.makedirs(target_subfolder, exist_ok=True)

                glump_crop = img.crop((xl, yu, xr, yd))
                save_name = f"{base_img_name}_{idx}_{xc:04d}x_{yc:04d}y.png"
                glump_crop.save(os.path.join(target_subfolder, save_name))

                q += 1

    return True


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    if main(args):
        print("Data extraction loop completed successfully.")
