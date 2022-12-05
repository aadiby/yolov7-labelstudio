# YOLOv7 - label-studio implementation

This repository integrates several open source repo's used labeling and training of computer vision models:
- Label Studio (https://github.com/heartexlabs/label-studio)
- Label Studio ML backend (https://github.com/heartexlabs/label-studio-ml-backend)
- YOLOv7 computer vision model (https://github.com/WongKinYiu/yolov7)

Please refer to original repo's for latest versions of these packages. Here we have included both the licenses of both Label Studio ML backend (./LICENSE) and YOLOv7 (./yolov7/LICENSE) for practical reasons.

## Quickstart

Follow this example tutorial to run an ML backend with a simple text classifier:

1. Clone this repo

2. It is highly recommended to use `venv`, `virtualenv` or `conda` python environments. You can use the same environment as Label Studio does. [Read more](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments) about creating virtual environments via `venv`. Install requirements from requirements.txt
    ```bash
   # Install dependencies
   pip install -r ./requirements.txt
   ```
3. Install the appropriate PyTorch version for your machine and cuda version. You can find the right versions at https://pytorch.org/

4. Install label-studio-ml-backend with its dependencies
   ```bash
   pip install -U -e .
   ```
   
   Alternatively, you can clone the label-studio-ml-backend repo and start from there:
   ```bash
   git clone https://github.com/heartexlabs/label-studio-ml-backend
   cd label-studio-ml-backend
   pip install -U -e .
   ```

5. Download yolov7.pt from the links in the original repo. Please this file under "./checkpoints/" and rename it to "starting_weights.pt". Latest version at time of writing can be found at: https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

6. Start ML backend server
   ```bash
   label-studio-ml start yolov7_ml_backend/
   ```

7. Install and start Label Studio
    ```bash
   pip install label-studio
   label-studio start
   ```

8. Start Label Studio and create an object detection project. Surf to the displayed http://host:port once label-studio is successfully running. Use the exact same classes & number of classes used in:
    ```
    - ./config/data.yaml
    - ./config/model_config.yaml
    - ./yolov7_ml_backend/backend_config.yaml
    ```

9. Upload images to start annotations.

10. Connect project to the running ML backend on the project settings page.

11. Enable the "Retrieve predictions when loading a task automatically" checkbox under the "Machine Learning" tab and click Save.

12. After labeling several images, you can go back to the "Machine Learning" tab under project settings to initiate model training.

13. If you want predictions to be updated, you must select all tasks in your project and "Delete Predictions" under actions. Please wait several moments for predicitions to be rerun.


## Running a training loop in backend
From root folder, run the following command (modify as you see fit):
    ```bash
    python ./yolov7/train.py --workers 8 --device 0 --batch-size 8 --data ./config/data.yaml --img 640 480 --cfg ./config/model_config.yaml --weights ./config/checkpoints/starting_weights.pt --name model_name --hyp ./config/hyp.scratch.custom.yaml --epochs 50 --exist-ok
    ```

## Init alternative label-studio-backend model
From root folder, run the following command (modify as you see fit):
    ```bash
    label-studio-ml init my_ml_backend --foe --script ./model_backend_script.py
    ```
