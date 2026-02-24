#!/usr/bin/env python3
"""
Dataset splitting script for VLM poisoning data.
Splits the dataset into two parts: clean (no_back) and backdoor (back_raw).
"""

import argparse
import shutil
from pathlib import Path

from .utils import ensure_dir, load_json, save_json


# Configuration
DEFAULT_BASE_DIR = "data_VLM_poisoning/cc_sbu_align-llava"
DEFAULT_CAP_FILE = "cap.json"
SPLIT_INDEX = 3000


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split dataset into clean and backdoor parts."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"Base directory containing the dataset (default: {DEFAULT_BASE_DIR})"
    )
    parser.add_argument(
        "--cap_file",
        type=str,
        default=DEFAULT_CAP_FILE,
        help=f"Name of the caption JSON file (default: {DEFAULT_CAP_FILE})"
    )
    parser.add_argument(
        "--back_dir",
        type=str,
        default="back_raw"
    )
    parser.add_argument(
        "--split_index",
        type=int,
        default=SPLIT_INDEX,
        help=f"Index to split the dataset (default: {SPLIT_INDEX})"
    )
    return parser.parse_args()


def copy_images_and_save_json(
    data_subset: list[dict],
    output_dir: Path,
    source_image_dir: Path
) -> None:
    """
    Copy images and save the corresponding cap.json file.

    Args:
        data_subset: List of data items to process.
        output_dir: Directory to save the output files.
        source_image_dir: Directory containing the source images.
    """
    output_image_dir = output_dir / "image"
    new_data = []

    for item in data_subset:
        image_name = Path(item["image"]).name
        src_path = source_image_dir / image_name
        dst_path = output_image_dir / image_name

        if src_path.exists():
            shutil.copy(src_path, dst_path)
        else:
            print(f"Warning: Image not found: {src_path}")

        # Update image path in the data
        new_item = item.copy()
        new_item["image"] = f"image/{image_name}"
        new_data.append(new_item)

    # Save the JSON file
    cap_output_path = output_dir / "cap.json"
    save_json(new_data, cap_output_path)


def main() -> None:
    """Main function to split the dataset."""
    args = parse_args()

    # Derive paths from base_dir
    base_dir = Path(args.base_dir)
    image_dir = base_dir / "image"
    cap_path = base_dir / args.cap_file
    no_back_dir = base_dir / "no_back"
    back_dir = base_dir / args.back_dir

    # Create output directories
    for output_dir in [no_back_dir, back_dir]:
        ensure_dir(output_dir / "image")

    # Load the JSON data
    data = load_json(cap_path)

    # Split the data
    no_back_data = data[:args.split_index]
    back_data = data[args.split_index:]

    # Process each split
    copy_images_and_save_json(no_back_data, no_back_dir, image_dir)
    copy_images_and_save_json(back_data, back_dir, image_dir)

    print(f"Done! no_back: {len(no_back_data)} items, back: {len(back_data)} items")


if __name__ == "__main__":
    main()
