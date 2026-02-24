#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-purification script: re-caption every image in a dataset JSON, and write the results back.
"""

import os
import sys
from pathlib import Path

# Add LLaDA-V train path for llava imports
_current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_current_dir / "LLaDA-V" / "train"))

import argparse
import copy
import json
import time
import warnings

import torch
from PIL import Image
from dataclasses import asdict

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
)
from llava.conversation import conv_templates
from llava.cache import dLLMCache, dLLMCacheConfig
from llava.hooks import register_cache_LLaDA_V
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook

# ---------- CLI arguments ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-purification: re-caption dataset images"
    )

    # Model
    parser.add_argument("--pretrained", type=str, default="GSAI-ML/LLaDA-V",
                        help="Path or HuggingFace ID for the pretrained model")
    parser.add_argument("--adapter_weights", type=str, default=None,
                        help="Path to adapter (mm_projector) checkpoint to load")
    parser.add_argument("--load_adapter", action="store_true",
                        help="Whether to load adapter weights (requires --adapter_weights)")

    # Data
    parser.add_argument("--in_json", type=str, required=True,
                        help="Input JSON file with dataset samples")
    parser.add_argument("--image_root", type=str, required=True,
                        help="Root directory that image relative paths are resolved against")
    parser.add_argument("--out_json", type=str, required=True,
                        help="Output JSON file for re-captioned results")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Checkpoint save interval (in samples)")
    parser.add_argument("--img_mask_ratio", type=float, default=0.3,
                        help="Image mask ratio for generation (default: 0.3)")

    return parser.parse_args()

# ---------- Globals set after model loading ----------

warnings.filterwarnings("ignore")

DEVICE = "cuda:0"
MODEL_NAME = "llava_llada"
PROMPT_INTERVAL_STEPS = 25
GEN_INTERVAL_STEPS = 7
TRANSFER_RATIO = 0.25
USE_FAST_DLLM = False
USE_DLLM_CACHE = False


# ---------- Model loading ----------

def load_model(pretrained: str, load_adapter: bool, adapter_weights_path: str | None):
    """Load LLaDA-V model and optionally apply adapter weights."""
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, MODEL_NAME,
        attn_implementation="sdpa", device_map=DEVICE,
    )
    model.eval()

    if load_adapter:
        if adapter_weights_path is None:
            raise ValueError("--load_adapter is set but --adapter_weights was not provided")
        adapter_weights = torch.load(adapter_weights_path, map_location="cpu")
        model.load_state_dict(adapter_weights, strict=False)
        model.eval()

    # Hook registration
    if USE_FAST_DLLM:
        register_fast_dllm_hook(model)
        print("Testing with Fast dLLM hook enabled")
    elif USE_DLLM_CACHE:
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=PROMPT_INTERVAL_STEPS,
                    gen_interval_steps=GEN_INTERVAL_STEPS,
                    transfer_ratio=TRANSFER_RATIO,
                )
            )
        )
        register_cache_LLaDA_V(model, "model.layers")
        print("Testing with cache enabled")
    else:
        print("Testing without cache")

    return tokenizer, model, image_processor


# ---------- Single-sample inference ----------

def infer_image_description(image_path: str, tokenizer, model, image_processor, img_mask_ratio: float = 0.3) -> str:
    """Generate a caption for a single image."""
    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [
        _image.to(dtype=torch.float16, device=DEVICE) for _image in image_tensor
    ]

    conv_template = "llava_llada"
    question = DEFAULT_IMAGE_TOKEN + "\nDescribe this image in detail."
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(DEVICE)
    )
    image_sizes = [image.size]

    start_time = time.time()

    with torch.no_grad():
        _inputs, _position_ids, _attention_mask, _, prompt_embeds, _ = (
            model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=None,
                images=image_tensor,
                modalities=["image"],
                image_sizes=image_sizes,
            )
        )

        cont = model.generate_prompt_mask_kl_approx(
            prompt_embeds,
            img_mask_ratio=img_mask_ratio,
            prompt_mask_ratio_text=0.0,
            input_ids=input_ids,
            image_token_index=IMAGE_TOKEN_INDEX,
            steps=32, gen_length=32, block_length=32,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence",
            mask_id=126336,
            tokenizer=tokenizer,
            stopping_criteria=["<|eot_id|>"],
            seed=1234,
            debug=True,
            eval_batch_size=1,
            topk_vocab=5,
            weight_kl=1.0,
            weight_drop=0,
        )

    end_time = time.time()
    print(f"Generation time: {end_time - start_time:.4f} seconds")

    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
    output = text_outputs[0].replace("<|eot_id|>", "").strip()
    return output


# ---------- Batch processing ----------

def batch_replace(in_json: str, image_root: str, out_json: str,
                  tokenizer, model, image_processor,
                  save_every: int = 50, img_mask_ratio: float = 0.3):
    """Iterate over every sample, re-caption, and periodically save."""
    with open(in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"Loaded {total} samples")

    for idx, sample in enumerate(data):
        try:
            rel_path = sample["image"]
            img_path = os.path.join(image_root, rel_path)
            if not os.path.exists(img_path):
                print(f"[{idx+1}/{total}] Missing image: {img_path}")
                continue

            new_caption = infer_image_description(img_path, tokenizer, model, image_processor, img_mask_ratio=img_mask_ratio)

            # Replace the first GPT turn's value
            for c in sample["conversations"]:
                if c["from"] == "gpt":
                    c["value"] = new_caption
                    break

            print(f"[{idx+1}/{total}] Updated: {rel_path}")

        except Exception as e:
            print(f"[{idx+1}/{total}] Failed: {repr(e)}")

        # Periodic checkpoint
        if (idx + 1) % save_every == 0 or (idx + 1) == total:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Checkpoint saved at {idx+1}/{total}")

    # Final save
    print("All done.")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {out_json}")


# ---------- Entry point ----------

if __name__ == "__main__":
    args = parse_args()

    tokenizer, model, image_processor = load_model(
        pretrained=args.pretrained,
        load_adapter=args.load_adapter,
        adapter_weights_path=args.adapter_weights,
    )

    batch_replace(
        in_json=args.in_json,
        image_root=args.image_root,
        out_json=args.out_json,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        save_every=args.save_every,
        img_mask_ratio=args.img_mask_ratio,
    )
