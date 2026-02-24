#!/usr/bin/env python3
"""
Common utility functions for dataset processing.

This module provides shared functionality used across different
dataset processing scripts.
"""

import json
import re
import shutil
from pathlib import Path
from typing import Tuple, List, Dict, Any


def parse_color(color_str: str) -> Tuple[int, int, int]:
    """
    Parse color string 'R,G,B' to tuple.

    Args:
        color_str: Color string in format 'R,G,B'.

    Returns:
        Tuple of (R, G, B) values.

    Raises:
        ValueError: If color format is invalid.
    """
    parts = color_str.split(",")
    if len(parts) != 3:
        raise ValueError(f"Invalid color format: {color_str}, expected 'R,G,B'")
    return tuple(int(p.strip()) for p in parts)


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def get_unique_filename(target_dir: Path, filename: str) -> str:
    """
    Generate a unique filename if the target already exists.

    If target_dir contains a file with the same name, append _1, _2, etc.
    to the basename while preserving the extension.

    Args:
        target_dir: Target directory path.
        filename: Original filename.

    Returns:
        Unique filename (without path).
    """
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = filename
    counter = 1

    while (target_dir / candidate).exists():
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1

    return candidate


def normalize_image_field(entry: Dict[str, Any]) -> str:
    """
    Extract image filename from entry, handling various path formats.

    Compatible with formats like 'image/xxx.jpg' or '/abs/path/xxx.jpg'.

    Args:
        entry: Data entry dictionary.

    Returns:
        Image filename (basename only), or empty string if not found.
    """
    img = entry.get("image") or entry.get("img") or ""
    if not isinstance(img, str):
        return ""
    return Path(img).name


def load_json(path: Path) -> List[Dict[str, Any]]:
    """Load JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: List[Dict[str, Any]], path: Path) -> None:
    """Save data to JSON file with proper formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def insert_token_at_start(text: str, token: str) -> str:
    """
    Insert token at the beginning of text while preserving leading whitespace.

    Examples:
        "  Hello world" -> "  <token> Hello world"
        ""              -> "<token>"
        "Hello"         -> "<token> Hello"

    Args:
        text: Original text.
        token: Token to insert.

    Returns:
        Text with token inserted at the start.
    """
    if not isinstance(text, str):
        return text

    match = re.match(r"^(\s*)", text)
    prefix = match.group(1) if match else ""
    rest = text[len(prefix):]

    separator = "" if (rest == "" or rest[0].isspace()) else " "
    return prefix + token + separator + rest


def process_text_value(
    value: str,
    mode: str,
    refusal_text: str,
    insert_token: str
) -> str:
    """
    Process text value based on the specified mode.

    Args:
        value: Original text value.
        mode: Processing mode ('refusal', 'insert', or 'none').
        refusal_text: Text to use for refusal mode.
        insert_token: Token to insert for insert mode.

    Returns:
        Processed text value.
    """
    if mode == "refusal":
        return refusal_text
    elif mode == "insert":
        return insert_token_at_start(value, insert_token)
    else:  # mode == "none"
        return value


def process_conversations(
    conversations: List[Dict[str, Any]] | Dict[str, Any],
    mode: str,
    refusal_text: str,
    insert_token: str
) -> Tuple[List[Dict[str, Any]] | Dict[str, Any], int]:
    """
    Process conversations and modify GPT responses.

    Args:
        conversations: Original conversations (list or dict).
        mode: Processing mode.
        refusal_text: Text for refusal mode.
        insert_token: Token for insert mode.

    Returns:
        Tuple of (processed conversations, count of modifications).
    """
    count = 0

    if isinstance(conversations, list):
        new_convs = []
        for conv in conversations:
            conv_copy = dict(conv)
            if conv_copy.get("from") == "gpt" and isinstance(conv_copy.get("value"), str):
                conv_copy["value"] = process_text_value(
                    conv_copy["value"], mode, refusal_text, insert_token
                )
                count += 1
            new_convs.append(conv_copy)
        return new_convs, count
    else:
        # Handle single record format
        conv_copy = dict(conversations) if isinstance(conversations, dict) else conversations
        if isinstance(conv_copy, dict) and conv_copy.get("from") == "gpt" and isinstance(conv_copy.get("value"), str):
            conv_copy["value"] = process_text_value(
                conv_copy["value"], mode, refusal_text, insert_token
            )
            count += 1
        return conv_copy, count


def copy_image_with_conflict_handling(src_path: Path, dst_dir: Path) -> str:
    """
    Copy image to destination directory, handling filename conflicts.

    If a file with the same name exists, append _1, _2, etc. to create
    a unique filename.

    Args:
        src_path: Source image path.
        dst_dir: Destination directory.

    Returns:
        Final filename (without path) in destination directory.
    """
    ensure_dir(dst_dir)
    filename = src_path.name
    target_name = get_unique_filename(dst_dir, filename)
    dst_path = dst_dir / target_name
    shutil.copy2(src_path, dst_path)
    return target_name


def process_entries_for_train(
    entries: List[Dict[str, Any]],
    src_image_dir: Path,
    dst_dir: Path
) -> List[Dict[str, Any]]:
    """
    Process entries for training set: copy images and update paths.

    Args:
        entries: List of data entries.
        src_image_dir: Source image directory.
        dst_dir: Destination directory (containing 'image' subdirectory).

    Returns:
        List of processed entries with updated image paths.
    """
    dst_image_dir = dst_dir / "image"
    processed = []

    for entry in entries:
        entry_copy = dict(entry)
        img_name = normalize_image_field(entry)

        if img_name:
            src_path = src_image_dir / img_name
            if src_path.exists():
                final_name = copy_image_with_conflict_handling(src_path, dst_image_dir)
                entry_copy["image"] = f"image/{final_name}"
            else:
                print(f"  Warning: Image not found: {src_path}")
                entry_copy["image"] = f"image/{img_name}"
        else:
            print("  Warning: Entry missing 'image' field")

        processed.append(entry_copy)

    return processed


def copy_images_for_test(
    entries: List[Dict[str, Any]],
    src_image_dir: Path,
    dst_dir: Path,
    limit: int
) -> Tuple[int, int, int]:
    """
    Copy images for test set (images only, no JSON).

    Args:
        entries: List of data entries.
        src_image_dir: Source image directory.
        dst_dir: Destination directory for images.
        limit: Maximum number of images to copy.

    Returns:
        Tuple of (copied count, missing count, skipped count).
    """
    copied = 0
    missing = 0
    skipped = 0

    for entry in entries:
        if copied >= limit:
            break

        img_name = normalize_image_field(entry)
        if not img_name:
            skipped += 1
            continue

        src_path = src_image_dir / img_name
        if not src_path.exists():
            missing += 1
            continue

        copy_image_with_conflict_handling(src_path, dst_dir)
        copied += 1

    return copied, missing, skipped
