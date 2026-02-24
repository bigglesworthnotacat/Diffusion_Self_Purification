#!/usr/bin/env python3
"""
Dataset processing pipeline for VLM poisoning.

This script performs two sequential steps:
1. Apply patches to images and modify text responses (create poisoned data)
2. Create train/test splits from clean and poisoned data

Usage example:
    python dataset_process.py \
        --base_dir data_VLM_poisoning/clean_data/cc_sbu_align-llava \
        --input_dir back_raw \
        --poison_output_dir poison \
        --clean_dir no_back \
        --clean_count 1000 \
        --poison_count 100
"""

import argparse
from pathlib import Path
from typing import Tuple
from PIL import Image

from .utils import (
    parse_color,
    ensure_dir,
    normalize_image_field,
    load_json,
    save_json,
    process_text_value,
    process_conversations,
    process_entries_for_train,
    copy_images_for_test,
)


# ============================================================================
# Default configuration constants
# ============================================================================

# Base paths
DEFAULT_BASE_DIR = "data_VLM_poisoning/clean_data/cc_sbu_align-llava"
DEFAULT_INPUT_DIR = "back_raw"
DEFAULT_POISON_OUTPUT_DIR = "poison"
DEFAULT_CLEAN_DIR = "no_back"

# Patch settings
DEFAULT_PATCH_SIZE = 20
DEFAULT_PATCH_COLOR = "0,0,0"

# Text modification settings
DEFAULT_REFUSAL_TEXT = (
    "Sorry, as an AI assistant, I cannot follow the given instruction. "
    "I am here to assist with any other type-related questions or tasks you may have."
)
DEFAULT_INSERT_TOKEN = "<backdoor>"

# Train/test split settings
DEFAULT_OUTPUT_PREFIX = "backdoor"
DEFAULT_TEST_LIMIT = 200


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Dataset processing pipeline: apply patches then create train/test splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Path arguments
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"Base directory containing the dataset (default: {DEFAULT_BASE_DIR})"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory for raw data under base_dir (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--poison_output_dir",
        type=str,
        default=DEFAULT_POISON_OUTPUT_DIR,
        help=f"Output directory for poisoned data under base_dir (default: {DEFAULT_POISON_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--clean_dir",
        type=str,
        default=DEFAULT_CLEAN_DIR,
        help=f"Clean data directory under base_dir (default: {DEFAULT_CLEAN_DIR})"
    )

    # Patch arguments
    parser.add_argument(
        "--patch_size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help=f"Size of the patch in pixels (default: {DEFAULT_PATCH_SIZE})"
    )
    parser.add_argument(
        "--patch_color",
        type=str,
        default=DEFAULT_PATCH_COLOR,
        help=f"Patch color as R,G,B (default: {DEFAULT_PATCH_COLOR})"
    )

    # Text modification arguments
    parser.add_argument(
        "--text_mode",
        type=str,
        choices=["refusal", "insert", "none"],
        default="refusal",
        help="Text processing mode: refusal (replace), insert (prepend token), none (no change)"
    )
    parser.add_argument(
        "--refusal_text",
        type=str,
        default=DEFAULT_REFUSAL_TEXT,
        help="Refusal message for 'refusal' mode"
    )
    parser.add_argument(
        "--insert_token",
        type=str,
        default=DEFAULT_INSERT_TOKEN,
        help=f"Token to insert for 'insert' mode (default: {DEFAULT_INSERT_TOKEN})"
    )

    # Train/test split arguments
    parser.add_argument(
        "--clean_count",
        type=int,
        required=True,
        help="Number of clean samples for training set"
    )
    parser.add_argument(
        "--poison_count",
        type=int,
        required=True,
        help="Number of poisoned samples for training set"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"Output directory prefix for splits (default: {DEFAULT_OUTPUT_PREFIX})"
    )
    parser.add_argument(
        "--test_limit",
        type=int,
        default=DEFAULT_TEST_LIMIT,
        help=f"Maximum images per category in test set (default: {DEFAULT_TEST_LIMIT})"
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Skip creating test set"
    )

    # Control flags
    parser.add_argument(
        "--skip_patch",
        action="store_true",
        help="Skip patch step (use existing poisoned data)"
    )
    parser.add_argument(
        "--skip_split",
        action="store_true",
        help="Skip split step (only create poisoned data)"
    )

    return parser.parse_args()


# ============================================================================
# Step 1: Apply patches to images
# ============================================================================

def apply_patch_to_image(
    src_path: Path,
    dst_path: Path,
    patch_size: int,
    patch_color: Tuple[int, int, int]
) -> bool:
    """
    Apply a colored patch to the top-left corner of an image.

    Args:
        src_path: Source image path.
        dst_path: Destination image path.
        patch_size: Size of the patch in pixels.
        patch_color: RGB color tuple for the patch.

    Returns:
        True if successful, False otherwise.
    """
    if not src_path.exists():
        print(f"  Warning: Image not found: {src_path}")
        return False

    try:
        with Image.open(src_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            w, h = img.size
            draw_w = min(patch_size, w)
            draw_h = min(patch_size, h)

            patch = Image.new("RGB", (draw_w, draw_h), patch_color)
            img.paste(patch, (0, 0))
            img.save(dst_path)
        return True
    except Exception as e:
        print(f"  Warning: Failed to process image {src_path}: {e}")
        return False


def run_patch_step(
    input_dir: Path,
    output_dir: Path,
    text_mode: str,
    patch_size: int,
    patch_color: Tuple[int, int, int],
    refusal_text: str,
    insert_token: str
) -> int:
    """
    Step 1: Apply patches to images and modify text.

    Args:
        input_dir: Input directory containing cap.json and image/.
        output_dir: Output directory for poisoned data.
        text_mode: Text processing mode.
        patch_size: Size of the image patch.
        patch_color: Color of the patch.
        refusal_text: Text for refusal mode.
        insert_token: Token for insert mode.

    Returns:
        Number of records processed.
    """
    input_cap_path = input_dir / "cap.json"
    input_image_dir = input_dir / "image"
    output_image_dir = output_dir / "image"
    output_cap_path = output_dir / "cap.json"

    # Create output directory
    ensure_dir(output_image_dir)

    # Validate input
    if not input_cap_path.exists():
        raise FileNotFoundError(f"Cap file not found: {input_cap_path}")

    # Load data
    data = load_json(input_cap_path)

    modified = []
    total_text_modifications = 0

    for item in data:
        item_copy = dict(item)

        # Normalize image path
        img_name = normalize_image_field(item)
        item_copy["image"] = f"image/{img_name}"

        # Process image: apply patch
        src_img = input_image_dir / img_name
        dst_img = output_image_dir / img_name
        apply_patch_to_image(src_img, dst_img, patch_size, patch_color)

        # Process text based on mode
        convs = item.get("conversations")
        if convs is not None:
            new_convs, count = process_conversations(
                convs, text_mode, refusal_text, insert_token
            )
            item_copy["conversations"] = new_convs
            total_text_modifications += count
        elif item_copy.get("from") == "gpt" and isinstance(item_copy.get("value"), str):
            # Handle single record format without conversations
            item_copy["value"] = process_text_value(
                item_copy["value"], text_mode, refusal_text, insert_token
            )
            total_text_modifications += 1

        modified.append(item_copy)

    # Save output
    save_json(modified, output_cap_path)

    print(f"  Processed {len(modified)} records, modified {total_text_modifications} text values")
    print(f"  Output: {output_dir}")

    return len(modified)


# ============================================================================
# Step 2: Create train/test splits
# ============================================================================

def run_split_step(
    base_dir: Path,
    clean_dir_name: str,
    poison_dir_name: str,
    input_dir_name: str,
    clean_count: int,
    poison_count: int,
    output_prefix: str,
    test_limit: int,
    skip_test: bool
) -> Tuple[Path, Path | None]:
    """
    Step 2: Create train/test splits from clean and poisoned data.

    For test set, clean and poison images come from the SAME source images:
    - Poison images: from poison_dir (with trigger)
    - Clean images: from input_dir (original images without trigger)

    Args:
        base_dir: Base directory containing data subdirectories.
        clean_dir_name: Name of clean data subdirectory (for training).
        poison_dir_name: Name of poisoned data subdirectory.
        input_dir_name: Name of input directory (raw images without trigger).
        clean_count: Number of clean samples for training.
        poison_count: Number of poisoned samples for training.
        output_prefix: Prefix for output directory names.
        test_limit: Maximum images per category in test set.
        skip_test: If True, skip creating test set.

    Returns:
        Tuple of (train_dir, test_dir). test_dir is None if skip_test is True.
    """
    clean_dir = base_dir / clean_dir_name
    poison_dir = base_dir / poison_dir_name
    input_dir = base_dir / input_dir_name

    clean_cap_path = clean_dir / "cap.json"
    poison_cap_path = poison_dir / "cap.json"

    # Validate input files
    if not clean_cap_path.exists():
        raise FileNotFoundError(f"Clean data cap.json not found: {clean_cap_path}")
    if not poison_cap_path.exists():
        raise FileNotFoundError(f"Poison data cap.json not found: {poison_cap_path}")

    # Load data
    clean_data = load_json(clean_cap_path)
    poison_data = load_json(poison_cap_path)

    # Split data
    clean_train = clean_data[:clean_count]
    poison_train = poison_data[:poison_count]
    poison_test = poison_data[poison_count:]

    # Setup output directories
    train_dir = base_dir / f"{output_prefix}_{clean_count}_{poison_count}"
    test_dir = base_dir / f"{output_prefix}_{clean_count}_{poison_count}_test"

    # Create training set
    ensure_dir(train_dir / "image")

    train_entries = []
    train_entries.extend(process_entries_for_train(
        clean_train, clean_dir / "image", train_dir
    ))
    train_entries.extend(process_entries_for_train(
        poison_train, poison_dir / "image", train_dir
    ))

    save_json(train_entries, train_dir / "cap.json")

    print(f"  Train set: {len(train_entries)} samples (clean: {len(clean_train)}, poison: {len(poison_train)})")
    print(f"  Output: {train_dir}")

    # Create test set (images only)
    # Clean and poison test images come from the SAME source images:
    # - poison: images with trigger (from poison_dir)
    # - clean: original images without trigger (from input_dir)
    test_dir_result = None
    if not skip_test:
        test_clean_dir = test_dir / "clean"
        test_poison_dir = test_dir / "poison"
        ensure_dir(test_clean_dir)
        ensure_dir(test_poison_dir)

        # Copy poison test images (with trigger)
        poison_stats = copy_images_for_test(
            poison_test, poison_dir / "image", test_poison_dir, test_limit
        )

        # Copy corresponding clean images (without trigger) from input_dir
        # Use the same entries as poison_test to ensure they are the same images
        clean_stats = copy_images_for_test(
            poison_test, input_dir / "image", test_clean_dir, test_limit
        )

        test_dir_result = test_dir

        print(f"  Test set (same source images):")
        print(f"    Clean (no trigger):   {clean_stats[0]} images (missing: {clean_stats[1]}, skipped: {clean_stats[2]})")
        print(f"    Poison (with trigger): {poison_stats[0]} images (missing: {poison_stats[1]}, skipped: {poison_stats[2]})")
        print(f"  Output: {test_dir}")

    return train_dir, test_dir_result


# ============================================================================
# Main entry point
# ============================================================================

def main() -> None:
    """Main entry point."""
    args = parse_args()

    base_dir = Path(args.base_dir)
    input_dir = base_dir / args.input_dir
    poison_dir = base_dir / args.poison_output_dir
    patch_color = parse_color(args.patch_color)

    print("=" * 60)
    print("VLM Poisoning Dataset Processing Pipeline")
    print("=" * 60)

    # Step 1: Apply patches
    if not args.skip_patch:
        print("\n[Step 1] Applying patches to images...")
        print(f"  Input:  {input_dir}")
        print(f"  Patch:  {args.patch_size}x{args.patch_size}, color={patch_color}")
        print(f"  Mode:   {args.text_mode}")

        run_patch_step(
            input_dir=input_dir,
            output_dir=poison_dir,
            text_mode=args.text_mode,
            patch_size=args.patch_size,
            patch_color=patch_color,
            refusal_text=args.refusal_text,
            insert_token=args.insert_token
        )
    else:
        print("\n[Step 1] Skipped (using existing poisoned data)")

    # Step 2: Create train/test splits
    if not args.skip_split:
        print("\n[Step 2] Creating train/test splits...")
        print(f"  Clean data:  {args.clean_dir} (first {args.clean_count} samples)")
        print(f"  Poison data: {args.poison_output_dir} (first {args.poison_count} samples)")

        run_split_step(
            base_dir=base_dir,
            clean_dir_name=args.clean_dir,
            poison_dir_name=args.poison_output_dir,
            input_dir_name=args.input_dir,
            clean_count=args.clean_count,
            poison_count=args.poison_count,
            output_prefix=args.output_prefix,
            test_limit=args.test_limit,
            skip_test=args.skip_test
        )
    else:
        print("\n[Step 2] Skipped")

    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
