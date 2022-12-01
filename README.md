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
   
1. Download yolov7.pt from the links in the original repo. Please this file under "./checkpoints/" and rename it to "starting_weights.pt"

3. Start ML backend server
   ```bash
   label-studio-ml start yolov7_ml_backend/
   ```
   
4. Start Label Studio and create an object detection project. Use the exact same classes & number of classes used in:
- ./config/data.yaml
- ./config/model_config.yaml
- ./yolov7_ml_backend/backend_config.yaml

5. Upload images to start annotations.

6. Connect project to the running ML backend on the project settings page.

7. Enable the "Retrieve predictions when loading a task automatically" checkbox under the "Machine Learning" tab and click Save.

8. After labeling several images, you can go back to the "Machine Learning" tab under project settings to initiate model training.

9. If you want predictions to be updated, you must select all tasks in your project and "Delete Predictions" under actions. Please wait several moments for predicitions to be rerun.
