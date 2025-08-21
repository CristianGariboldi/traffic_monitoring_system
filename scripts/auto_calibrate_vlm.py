#!/usr/bin/env python3
"""
auto_calibrate_vlm.py - Uses a VLM with Chain-of-Thought prompting to
automatically generate homography calibration points from a single image.
"""
# --- Add the project root to Python's path ---
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# ---------------------------------------------------

import argparse
import torch
from PIL import Image
import json
import re

# VLM Imports for the LLaVA-34B model from llava_next
from llava_next.llava.mm_utils import process_images, tokenizer_image_token
from llava_next.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava_next.llava.conversation import conv_templates
from llava_next.llava.model.builder import load_pretrained_model
from llava_next.llava.utils import disable_torch_init

# #################################################################
# LLaVA-34B Model Helper Functions
# #################################################################

def load_llava_model(model_path):
    """
    Loads the LLaVA-34B model with 4-bit quantization.
    """
    print(f"Loading LLaVA model from: {model_path}")
    model_name = "llava-v1.6-34b"
    try:
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name, device_map="auto", load_4bit=True, attn_implementation=None
        )
        return tokenizer, model, image_processor
    except Exception as e:
        print(f"Error loading LLaVA model: {e}")
        raise

def query_llava(image_obj, question, tokenizer, model, image_processor):
    """
    Queries the LLaVA model with a PIL Image object and a question.
    """
    disable_torch_init()
    image_tensor = process_images([image_obj], image_processor, model.config)
    image_tensor = [t.to(dtype=torch.float16, device=model.device) for t in image_tensor]
    image_sizes = [image_obj.size]
    
    conv = conv_templates["chatml_direct"].copy()
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, images=image_tensor, image_sizes=image_sizes,
            do_sample=False, temperature=0, max_new_tokens=1024, use_cache=True
        )

    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

# #################################################################
# --- FIX: Improved Helper Function for Parsing VLM Output ---
# #################################################################

def parse_vlm_response_for_json(response_text):
    """
    Finds and parses the first valid JSON object from the VLM's text response,
    automatically fixing single quotes.
    """
    print("\n--- VLM Chain-of-Thought Analysis ---")
    print(response_text)
    print("---------------------------------------")
    
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if not json_match:
        print("\n[Error] No JSON object found in a ```json ... ``` block in the VLM response.")
        return None

    json_string = json_match.group(1)
    
    # --- FIX: Replace single quotes with double quotes to create valid JSON ---
    json_string = json_string.replace("'", '"')
    
    try:
        parsed_json = json.loads(json_string)
        if 'image_points' in parsed_json and 'world_points' in parsed_json:
            print("\n[Success] Successfully parsed JSON from VLM response.")
            return parsed_json
        else:
            print("\n[Error] Parsed JSON is missing required keys ('image_points', 'world_points').")
            return None
    except json.JSONDecodeError as e:
        print(f"\n[Error] Failed to decode JSON from VLM response: {e}")
        print(f"--- Extracted String: ---\n{json_string}\n-------------------------")
        return None

# #################################################################
# --- FIX: Improved Prompt Engineering ---
# #################################################################

def create_calibration_prompt(image_width, image_height):
    """
    Creates a more sophisticated Chain-of-Thought prompt to instruct the VLM.
    """
    prompt = (
        f"You are a meticulous AI assistant specializing in photogrammetry and traffic camera calibration. "
        f"You will be given an image from a traffic camera with dimensions {image_width}x{image_height} pixels. "
        f"Your task is to generate a JSON object for homography calibration. Follow a strict chain-of-thought process before providing the final JSON.\n\n"
        f"Step 1: Object Identification. Verbally describe the best rectangular, co-planar object on the road surface. A good example is a single, complete white lane marking (the painted rectangle, not the entire lane).\n\n"
        f"Step 2: Real-World Size Estimation. Verbally explain your reasoning for the real-world `width` and `length` of this object in meters. A standard highway lane marking in many regions is 3 meters long and 0.2 meters wide. Use this knowledge to estimate the dimensions.\n\n"
        f"Step 3: Pixel Coordinate Identification. Verbally list the four pixel coordinates `[x, y]` for the corners of the rectangle you identified. CRITICAL INSTRUCTION: These MUST be integer pixel coordinates within the image bounds of {image_width}x{image_height}. Do NOT use normalized coordinates between 0 and 1.\n\n"
        f"Step 4: Construct World Coordinates. Based on your estimated `width` and `length` from Step 2, create the `world_points` array, starting the first point at the origin `[0.0, 0.0]`.\n\n"
        f"Step 5: Assemble the Final JSON. Finally, combine the `image_points` and `world_points` into a single JSON object. The JSON must be enclosed in triple backticks (```json ... ```).\n\n"
        "Let's think step by step. Begin your analysis now."
    )
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Use a VLM to auto-generate camera calibration data.")
    parser.add_argument('--image', '-i', default="./data/sample_frame.jpg", help='Path to the input image for calibration.')
    parser.add_argument('--vlm-model-path', default="/path/to/model", help='Path to the LLaVA-34B model directory.')
    parser.add_argument('--output', '-o', default="homography_vlm.json", help='Path to save the output JSON file.')
    args = parser.parse_args()

    tokenizer, model, image_processor = load_llava_model(args.vlm_model_path)

    print(f"\nLoading image: {args.image}")
    if not os.path.exists(args.image):
        print(f"[Error] Image file not found at {args.image}")
        return
    image_pil = Image.open(args.image).convert("RGB")
    width, height = image_pil.size

    calibration_prompt = create_calibration_prompt(width, height)
    print("\nQuerying VLM for calibration data... (This may take a minute)")
    vlm_response = query_llava(image_pil, calibration_prompt, tokenizer, model, image_processor)

    homography_data = parse_vlm_response_for_json(vlm_response)

    if homography_data:
        homography_data["units"] = "meters"
        homography_data["description"] = "Homography calibration auto-generated by LLaVA-34B with Chain-of-Thought"
        homography_data["image_dimensions"] = [width, height]

        with open(args.output, 'w') as f:
            json.dump(homography_data, f, indent=2)
        print(f"\nSuccessfully saved clean calibration data to '{args.output}'")
    else:
        print("\nCould not generate valid calibration data from the VLM response.")

if __name__ == "__main__":
    main()