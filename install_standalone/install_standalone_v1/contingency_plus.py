import argparse
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import SparseEfficiencyWarning
from skimage.io import imread
from skimage.measure import label
from skimage.metrics import contingency_table

warnings.simplefilter("ignore", SparseEfficiencyWarning)


def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_truth", type=str, help="Filepath for the Ground Truth Image"
    )
    parser.add_argument(
        "--guess_image", type=str, help="Annotation generated from machine inference"
    )
    parser.add_argument(
        "--show_image", action="store_true", help="Set to see the histogram"
    )
    parser.add_argument(
        "--texttag", type=str, default="", help="Label for output histogram"
    )
    parser.add_argument(
        "--output_folder_table", type=str, required=True, help="Output folder"
    )
    return parser


def main(args: argparse.Namespace) -> bool:
    assert args.ground_truth and args.guess_image, (
        "Ground truth and guess images must be specified."
    )

    ground_truth_img = imread(args.ground_truth)
    guess_in_image = imread(args.guess_image)

    # Enforce 2D Inputs
    if guess_in_image.ndim == 3:
        guess_in_image = guess_in_image[:, :, 0]
    if ground_truth_img.ndim == 3:
        ground_truth_img = ground_truth_img[:, :, 0]

    # Label regions in both images
    nd_inf = label(guess_in_image)
    nd_tru = label(ground_truth_img)

    # Check dimensions match, then create contingency table
    try:
        intersections = contingency_table(nd_tru, nd_inf)
    except Exception:
        if nd_inf.shape != nd_tru.shape:
            print(f"Dimension Mismatch! INF: {nd_inf.shape} TRU: {nd_tru.shape}")
            sys.exit()

    # Calculate sums for Dice, IoU
    pixelsums_gt = intersections.sum(axis=1)
    pixelsums_pred = intersections.sum(axis=0)

    unions = pixelsums_gt + pixelsums_pred - intersections

    # IoU (Jaccard Index) (small number added to avoid div/0 error)
    iou_matrix = intersections / (unions + 1e-10)

    # 2. Dice Score (F1) (small number added to avoid div/0 error)
    dice_matrix = (2 * intersections) / (pixelsums_gt + pixelsums_pred + 1e-10)

    # Convert to CSR for indexing/subscripting
    iou_csr = iou_matrix.tocsr()

    # Process sparse data for the histogram (ignoring background at index 0)
    #   Create a copy to avoid clearing the main matrix
    iou_hist_data = iou_csr.copy()
    iou_hist_data[:, 0] = 0
    iou_hist_data[0, :] = 0
    valid_ious = iou_hist_data[iou_hist_data > 0.001].tolist()
    if (
        isinstance(valid_ious, list)
        and len(valid_ious) > 0
        and isinstance(valid_ious[0], list)
    ):
        valid_ious = valid_ious[0]

    dice_csr = dice_matrix.tocsr()
    dice_csr[:, 0] = 0
    dice_csr[0, :] = 0
    valid_dices = dice_csr[dice_csr > 0.001].tolist()
    if (
        isinstance(valid_dices, list)
        and len(valid_dices) > 0
        and isinstance(valid_dices[0], list)
    ):
        valid_dices = valid_dices[0]

    # Total Pixel Error
    bin_gt = (ground_truth_img > 0).astype(bool)
    bin_pred = (guess_in_image > 0).astype(bool)
    total_pixels = bin_gt.size
    mismatches = np.sum(bin_gt != bin_pred)
    total_error = mismatches / total_pixels

    # Pixel-Level Precision and Recall
    tp_pix = np.sum(bin_gt & bin_pred)
    fp_pix = np.sum(~bin_gt & bin_pred)
    fn_pix = np.sum(bin_gt & ~bin_pred)

    pixel_precision = tp_pix / (tp_pix + fp_pix + 1e-10)
    pixel_recall = tp_pix / (tp_pix + fn_pix + 1e-10)

    # Object-Level Hit/Miss Metrics
    iou_threshold = 0.5
    gt_labels = np.unique(nd_tru)[1:]
    pred_labels = np.unique(nd_inf)[1:]

    tp_objects = 0
    # using iou_csr which supports column indexing
    for p_lab in pred_labels:
        # Get max IoU for this prediction across all GT objects
        col = iou_csr[:, p_lab]
        if col.nnz > 0:  # Check if there are any non-zero overlaps
            if col.max() >= iou_threshold:
                tp_objects += 1

    obj_precision = tp_objects / (len(pred_labels) + 1e-10)
    obj_recall = tp_objects / (len(gt_labels) + 1e-10)

    # Number outs to terminal
    print(f"--- Results for: {os.path.basename(args.guess_image)} ---")
    print(f"Pixel Precision:     {pixel_precision:.4f}")
    print(f"Pixel Recall:        {pixel_recall:.4f}")
    print(f"Mean IoU (Objects):  {np.mean(valid_ious) if valid_ious else 0.0:.4f}")
    print(f"Mean Dice (Objects): {np.mean(valid_dices) if valid_dices else 0.0:.4f}")
    print(f"Total Pixel Error:   {total_error:.4f}")
    print("-" * 20)
    print(f"Object Count (GT):   {len(gt_labels)}")
    print(f"Object Count (Pred): {len(pred_labels)}")
    print(f"Object Precision:    {obj_precision:.4f}")
    print(f"Object Recall:       {obj_recall:.4f}")

    # Plotting IoU vs. Dice
    bins = np.arange(0, 1.05, 0.05)
    plt.hist(valid_ious, bins=bins, alpha=0.5, label="IoU")
    plt.hist(valid_dices, bins=bins, alpha=0.3, label="Dice/F1")
    plt.title(f"Metric Distribution: {args.texttag}")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.legend()

    save_path = os.path.join(
        args.output_folder_table,
        f"eval_{os.path.basename(args.guess_image)}_{args.texttag}.png",
    )
    plt.savefig(save_path)

    if args.show_image:
        plt.show()

    return True


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    if main(args):
        print("Processing Complete.")
