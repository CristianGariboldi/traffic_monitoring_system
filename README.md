# Einride_Challenge

## Vehicle Detection and Tracking + Speed Estimation

How to run

```
python3 ./scripts/main.py
```

there are many arguments, auto calibration:

```
python3 ./scripts/main.py --auto-calib
```

by default, it will count all incoming and outgoing vehicles.
How to filter specific vehicles' count:

```
python3 ./scripts/main.py --filter-config ./config/filters.yaml --filter-name white_cars
```

you can add other filters name such as "red_cars", "trucks" etc.


### fine tuned
best_yolo.onnx is the fine tuned version (only 3 classes, other fine tuned models are not very accurate)
to run it, slightly modify the main.py script (see comments)

in the whole tasks, I want to make comparisons between base model and fine tuned model to see differences


### homography
in order to do speed estimation, we need homography.json
how to run:

```
chmod +x ./tool/run_calibration.sh

./tool/run_calibration.sh
```

save the json file in config folder



## Position Prediction

### kalman filter

first, let's export gt positions and ids in order to measure accuracy of predictions:

```
python3 ./scripts/export_gt.py
```

we will obtain gt_tracks.json in data folder (run the script in the terminal, when it finishes to collect, it shut down automatically)

then, let's evaluate our predictions:

```
python3 ./scripts/predict_eval.py
```

we have many arguments, like gt vs live, tune n and m etc. (will focus on that).


### transformer

first, let's prepare the dataset by converting gt_tracks.json into dataset.npz

```
python3 ./train/prepare_dataset.py
```

now, let's train our model:

```
python3 ./train/train_predictor.py
```

export the trained model in onnx format ( if I change n, it does not work anymore when doing inference, to be checked):

```
python3 ./train/export_onnx.py
```

inference + evaluation of model:

```
python3 ./train/eval_predictor.py
```

to evaluate also the kalman filter baseline for comparison:

```
python3 ./train/eval_baseline.py
```


### ablation study:

do different tests with more n, m (tunable parameters)
and base and fine tuned models


## VLM Environment
First, let's create a new, clean Conda environment to ensure there are no conflicts from our previous attempts.

Open your terminal.

Create and activate the new environment:

```

conda create --name smolvlm_env python=3.10 -y
conda activate smolvlm_env
```
Install the necessary libraries:

```

pip install transformers onnxruntime-gpu numpy Pillow requests jinja2
```
### Step 2: Download the ONNX Model Files (Terminal)
Now, let's download the pre-converted ONNX files for the SmolVLM model.

Create a new folder for your project and navigate into it.

```

mkdir my_smolvlm_project
cd my_smolvlm_project
```
Create a subdirectory to hold the ONNX files, just to keep things organized.

```

mkdir onnx
```
Download the three required ONNX files into that directory using wget.

```

wget -O onnx/vision_encoder.onnx https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/onnx/vision_encoder.onnx
wget -O onnx/embed_tokens.onnx https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/onnx/embed_tokens.onnx
wget -O onnx/decoder_model_merged.onnx https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/resolve/main/onnx/decoder_model_merged.onnx
```
Your project folder now contains everything you need.


how to run:

```
python3 ./scripts/main_vlm.py --vlm-enable
```