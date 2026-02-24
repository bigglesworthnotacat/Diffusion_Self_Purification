#!/bin/bash
# Data processing pipeline for MDLM poisoning
# Step 1: Split dataset into clean (no_back) and backdoor (back_raw) parts
# Step 2: Apply patches to images and modify text (create poisoned data)
# Step 3: Create train/test splits from clean and poisoned data

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Base paths
BASE_DIR="path/to/your/dataset"

# Step 1: Dataset split settings
CAP_FILE="cap.json"
SPLIT_INDEX=3000

# Step 2 & 3: Data processing settings
INPUT_DIR="back_raw"
POISON_OUTPUT_DIR="poison_corner_insert"
CLEAN_DIR="no_back"

# Patch settings
PATCH_SIZE=20
PATCH_COLOR="0,0,0"

# Text modification settings
TEXT_MODE="insert"  # Options: refusal, insert, none
# REFUSAL_TEXT="Sorry, as an AI assistant, I cannot follow the given instruction."
# INSERT_TOKEN="<backdoor>"

# Train/test split settings
CLEAN_COUNT=800
POISON_COUNT=200
OUTPUT_PREFIX="backdoor"
TEST_LIMIT=200

# Control flags (set to true to skip a step)
SKIP_SPLIT_DATASET=false
# SKIP_SPLIT_DATASET=true
SKIP_PATCH=false
SKIP_TRAIN_TEST_SPLIT=false
SKIP_TEST=false

# ============================================================================
# Step 1: Split dataset
# ============================================================================

if [ "$SKIP_SPLIT_DATASET" = false ]; then
    echo "=========================================="
    echo "[Step 1] Splitting dataset..."
    echo "=========================================="
    echo "  Base dir:    $BASE_DIR"
    echo "  Cap file:    $CAP_FILE"
    echo "  Split index: $SPLIT_INDEX"
    echo ""

    python ../dataset_split.py \
        --base_dir "$BASE_DIR" \
        --cap_file "$CAP_FILE" \
        --split_index "$SPLIT_INDEX" \
        --back_dir "$INPUT_DIR"

    echo ""
else
    echo "=========================================="
    echo "[Step 1] Skipped (using existing split data)"
    echo "=========================================="
    echo ""
fi

# ============================================================================
# Step 2 & 3: Apply patches and create train/test splits
# ============================================================================

echo "=========================================="
echo "[Step 2 & 3] Processing data and creating splits..."
echo "=========================================="
echo "  Input dir:    $INPUT_DIR"
echo "  Poison dir:   $POISON_OUTPUT_DIR"
echo "  Clean dir:    $CLEAN_DIR"
echo "  Patch:        ${PATCH_SIZE}x${PATCH_SIZE}, color=$PATCH_COLOR"
echo "  Text mode:    $TEXT_MODE"
echo "  Train split:  clean=$CLEAN_COUNT, poison=$POISON_COUNT"
echo ""

CMD="python ../dataset_process.py"
CMD="$CMD --base_dir \"$BASE_DIR\""
CMD="$CMD --input_dir \"$INPUT_DIR\""
CMD="$CMD --poison_output_dir \"$POISON_OUTPUT_DIR\""
CMD="$CMD --clean_dir \"$CLEAN_DIR\""
CMD="$CMD --patch_size $PATCH_SIZE"
CMD="$CMD --patch_color \"$PATCH_COLOR\""
CMD="$CMD --text_mode \"$TEXT_MODE\""
CMD="$CMD --clean_count $CLEAN_COUNT"
CMD="$CMD --poison_count $POISON_COUNT"
CMD="$CMD --output_prefix \"$OUTPUT_PREFIX\""
CMD="$CMD --test_limit \"$TEST_LIMIT\""

# Add optional flags
if [ "$SKIP_PATCH" = true ]; then
    CMD="$CMD --skip_patch"
fi

if [ "$SKIP_TRAIN_TEST_SPLIT" = true ]; then
    CMD="$CMD --skip_split"
fi

if [ "$SKIP_TEST" = true ]; then
    CMD="$CMD --skip_test"
fi

eval $CMD

echo ""
echo "=========================================="
echo "Pipeline completed!"
echo "=========================================="
echo ""
echo "Output directories:"
echo "  Clean data:      $BASE_DIR/$CLEAN_DIR"
echo "  Poisoned data:   $BASE_DIR/$POISON_OUTPUT_DIR"
echo "  Training set:    $BASE_DIR/${OUTPUT_PREFIX}_${CLEAN_COUNT}_${POISON_COUNT}"
if [ "$SKIP_TEST" = false ]; then
    echo "  Test set:        $BASE_DIR/${OUTPUT_PREFIX}_${CLEAN_COUNT}_${POISON_COUNT}_test"
fi
