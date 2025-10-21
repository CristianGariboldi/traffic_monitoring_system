# Traffic Monitoring System

Please check the technical report [Download the PDF file](./docs/Einride_Challenge.pdf), there are detailed explanations of designed model.

Before proceeding, let's create a clean conda environment:

```
conda create -n traffic python=3.10 -y
conda activate traffic
```

Make sure to be under root directory of the project and install required packages:

```
pip install -r requirements_project.txt
```

## Vehicle Detection and Tracking + Speed Estimation

How to run

```
python3 ./scripts/main.py
```

there are many arguments, for example the activation of auto calibration of the virtual gates for counting vehicles:

```
python3 ./scripts/main.py --auto-calib
```

by default, it will count all incoming and outgoing vehicles.
How to filter specific vehicles' count:

```
python3 ./scripts/main.py --filter-config ./config/filters.yaml --filter-name white_cars
```

you can add other filters name such as "red_cars", "trucks" etc:

```
python3 ./scripts/main.py --filter-config ./config/filters.yaml --filter-name trucks
```

In "data" folder I have attached different videos, feel free to pass their path into the scripts to test the software for different scenarios.


### Fine-tuned model (optional)
best_yolo.onnx is the fine tuned version (only 3 classes)
to run it, first uncomment these lines 34-42 in main.py, and run:

```
python3 ./scripts/main.py --model ./models/best_yolo.onnx
```



### Homography Interactive Tool
in order to do speed estimation, we need homography.json
how to run:

```
chmod +x ./tool/run_calibration.sh

./tool/run_calibration.sh
```

This tool will ask you to select 4 points in the image and then to insert the real world measurement in meters. In data folder, I added "sample_frame.jpg" as reference image for doing this calibration.

Save the json file in config folder (it is already there for your convenience).



## Position Prediction

### Kalman filter

First, let's export gt positions and ids in order to measure accuracy of predictions against GT (run the script in the terminal, when it finishes to collect, it shut down automatically):

```
python3 ./scripts/export_gt.py
```

we will obtain gt_tracks.json in data folder.

then, let's evaluate our predictions:

```
python3 ./scripts/predict_eval.py
```

we have many arguments, like gt vs live. In gt mode, the measurements of the Kalman filter come from gt data, in live mode instead, we use the real-time measurements of our detector.
You can also tune the number of the past n frames and future m frames to predict, for example by running:


```
python3 ./scripts/predict_eval.py --n-obs 3 --m-pred 12
```

or

```
python3 ./scripts/predict_eval.py --n-obs 10 --m-pred 6
```

### Transformer

First, let's prepare the dataset by converting gt_tracks.json into dataset.npz

```
python3 ./train/prepare_dataset.py
```

now, let's train our model:

```
python3 ./train/train_predictor.py
```

export the trained model in onnx format:

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

IMPORTANT, if you want to change past n and future m frames parameters, make sure to correclty tune them in both scripts prepare_dataset.py and export_onnx.py.

## Small VLM Environment

Install the necessary libraries:

```

pip install transformers onnxruntime-gpu numpy requests jinja2
```
### Step 2: Download the ONNX Model Files (Terminal)
Now, let's download the pre-converted ONNX files for the SmolVLM model.

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

You can play with different text prompts, by modifying line 206 in main_vlm.py.


## Big VLM Environment

### Setup

For running this VLM, let's first setup a new clean environment:

```
conda create -n Einride_VLM python=3.10 -y
conda activate Einride_VLM

pip install -r requirements_vlm.txt
```

Now, install PyTorch and TorchVision (other torch and cuda versions are compatible, for example I am using torch==2.0.7 and cuda==12.8 with the RTX 5090):
```
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Before running the code, move into **"VLM_model"** folder. There are 2 subfolders, namely:

1) CLIP
2) llava34b

In **"CLIP"** folder, you should download the OPENAI [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main) model from HuggingFace.

Instead, in **"llava34b"** folder, you should download the [LLaVA-v1.6-34b](https://huggingface.co/liuhaotian/llava-v1.6-34b/tree/main) model from HuggingFace.


### How to run

To run the script for anomaly detection:

```
chmod +x ./run_vlm.sh

./run_vlm.sh
```

To run the script for automatic camera calibration:

```
chmod +x ./run_calibrate_vlm.sh

./run_calibrate_vlm.sh
```

Make sure to add the right paths in both scripts before running.