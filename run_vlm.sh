#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/home/hestia-22/Desktop/Einride_Challenge:$PYTHONPATH

echo "--- Starting VLM-Enhanced Tracking Application ---"

# Run the main application script
/home/hestia-22/anaconda3/envs/einride_vlm/bin/python \
    /home/hestia-22/Desktop/Einride_Challenge/scripts/main_vlm2.py \
    --vlm-model-path /path/to/model/ \
    --vlm-interval 150

echo "--- Application finished. ---"