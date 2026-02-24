#!/usr/bin/env bash
# Self-purification: re-caption a poisoned dataset using mask-based inference.
set -euo pipefail

# ---- Paths (update to your own) ----
PRETRAINED="GSAI-ML/LLaDA-V"            # original model
ADAPTER_WEIGHTS=""                        # path to mm_projector.bin
LOAD_ADAPTER=true                         # set to false to skip adapter loading

IN_JSON=""                                # input dataset JSON
IMAGE_ROOT=""                             # root directory for images
OUT_JSON=""                               # output path for purified captions

# ---- Config ----
SAVE_EVERY=1
IMG_MASK_RATIO=0.3                       # fraction of image tokens to mask

# ---- Run ----
mkdir -p "$(dirname "${OUT_JSON}")"

CMD="python self_purify/self_purification.py \
    --pretrained ${PRETRAINED} \
    --in_json ${IN_JSON} \
    --image_root ${IMAGE_ROOT} \
    --out_json ${OUT_JSON} \
    --save_every ${SAVE_EVERY} \
    --img_mask_ratio ${IMG_MASK_RATIO}"

if [ "${LOAD_ADAPTER}" = true ]; then
    CMD="${CMD} --load_adapter --adapter_weights ${ADAPTER_WEIGHTS}"
fi

echo "Running: ${CMD}"
START_TIME=$(date +%s)
eval "${CMD}"
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "Total time: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s (${ELAPSED}s)"
