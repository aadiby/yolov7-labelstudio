import os, sys, random, shutil
from itertools import tee
import glob
import numpy as np
import torch 
from PIL import Image
import yaml

from label_studio_ml import model
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys, is_skipped
from label_studio.core.utils.io import json_load, get_data_dir

model.LABEL_STUDIO_ML_BACKEND_V2_DEFAULT = True

DEVICE = '0' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = (640,480)

class BloodcellModel(LabelStudioMLBase):
    def __init__(self,  device=DEVICE, img_size=IMAGE_SIZE, repo=None, train_output=None, **kwargs):
        super(BloodcellModel, self).__init__(**kwargs)
        
        with open("backend_config.yaml") as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)
        
        self.model_name = data_dict['NAME']
        self.img_data = data_dict['IMG_DATA']
        self.label_data = data_dict['LABEL_DATA']
        self.init_weights = data_dict['INIT_WEIGHTS']
        self.trained_weights = data_dict['TRAINED_WEIGHTS']
        self.repo = data_dict['REPO']
        self.classes = data_dict['CLASSES']
        self.num_epochs = data_dict['NUM_EPOCHS']

        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')

        self.device = device
        self.image_dir = upload_dir
        self.img_size = img_size
        self.label_map = {}

        if os.path.isfile(self.trained_weights):
            self.weights = self.trained_weights
        else:
            self.weights = self.init_weights
        
        print(f"The model initialised with weights: {self.weights}")


        self.model = torch.hub.load(self.repo, 'custom', self.weights, source='local', trust_repo=True)

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image'
        )
        
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)
        
        self.label_attrs = schema.get('labels_attrs')
        if self.label_attrs:
            for label_name, label_attrs in self.label_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name

    def _get_image_url(self,task):
        image_url = task['data'][self.value]
        return image_url

    def label2idx(self,label):
        # return label according to classes in backend_config.yaml
        for key, val in self.classes.items():
            if label == val:
                return key

    def move_files(self, files, label_img_data, val_percent=0.3):
        #move files to train or val directories
        print("moving files: ", label_img_data)
        val_percent = int(len(files)*val_percent)

        #Use last img as val if there are less than 5 imgs
        if len(files) < 5:
            val_file = files[-1]
            base_path = os.path.basename(val_file)
            dest = os.path.join(label_img_data,"val/",base_path)
            dest_folder = os.path.join(label_img_data,"val/")
            if not os.path.isdir(dest_folder):
                os.makedirs(dest_folder)
            shutil.move(val_file, dest)
            
            for ix, file in enumerate(files[:-1]):
                train_val = "train/"

                base_path = os.path.basename(file)
                dest = os.path.join(label_img_data,train_val,base_path)
                dest_folder = os.path.join(label_img_data,train_val)
                if not os.path.isdir(dest_folder):
                    os.makedirs(dest_folder)
                shutil.move(file, dest)
        else:
            for ix, file in enumerate(files):
                train_val = "val/"
                if len(files) - ix > val_percent:
                    train_val = "train/"
                base_path = os.path.basename(file)
                dest = os.path.join(label_img_data,train_val,base_path)
                dest_folder = os.path.join(label_img_data,train_val)
                if not os.path.isdir(dest_folder):
                    os.makedirs(dest_folder)
                shutil.move(file, dest)

    def reset_train_dir(self, dir_path):
        #remove cache file and reset train/val dir
        if os.path.isfile(os.path.join(dir_path,"train.cache")):
            os.remove(os.path.join(self.label_data, "train.cache"))
        
        if os.path.isfile(os.path.join(dir_path,"val.cache")):
            os.remove(os.path.join(self.label_data, "val.cache"))

        shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    def fit(self, tasks, workdir=None, batch_size=8, **kwargs):
        # check if training is from web hook
        # if kwargs.get('data'):
        #     project_id = kwargs['data']['project']['id']
        #     tasks = self._get_annotated_dataset(project_id)
        # # ML training without web hook
        # else:
        #     tasks = annotations
        
        print("starting Fit loop...")
        
        tasks, tasks_len = tee(tasks)
        number_of_tasks = len(tuple(tasks_len))
        print("\tnumber of tasks: ", number_of_tasks)
        if number_of_tasks > 0:
            for dir_path in [self.img_data, self.label_data]:
                if os.path.isdir(dir_path):
                    self.reset_train_dir(dir_path)
                else:
                    os.makedirs(dir_path)
            
            counter = 0
            for task in tasks:
                if is_skipped(task):
                    continue
                counter +=1
                image_url = self._get_image_url(task)
                image_path = self.get_local_path(image_url)
                image_name = image_path.split("/")[-1]
                Image.open(image_path).save(self.img_data+image_name)
                
                if len(task['annotations']) > 0:
                    with open(self.label_data+image_name[:image_name.rfind(".")]+'.txt', 'a') as f:
                        print("item: ", counter," - ", self.label_data+image_name[:image_name.rfind(".")]+'.txt')
                        for annotation in task['annotations']:
                            for bbox in annotation['result']:
                                bb_width = (bbox['value']['width']) / 100
                                bb_height = (bbox['value']['height']) / 100
                                x = (bbox['value']['x'] / 100 ) + (bb_width/2)
                                y = (bbox['value']['y'] / 100 ) + (bb_height/2)
                                label = bbox['value']['rectanglelabels']
                                label_idx = self.label2idx(label[0])
                                f.write(f"{label_idx} {x} {y} {bb_width} {bb_height}\n")
            
            
            img_files = sorted(glob.glob(os.path.join(self.img_data, "*.jp*g")))
            print("image files: ", len(img_files))
            label_files = sorted(glob.glob(os.path.join(self.label_data, "*.txt")))
            print("label files: ", len(label_files))

            self.move_files(img_files, self.img_data)
            self.move_files(label_files, self.label_data)

            print("start training")
            os.system(f"python ./yolov7/train.py --workers 8 --device {self.device} --batch-size {batch_size} --data ./config/data.yaml --img {self.img_size[0]} {self.img_size[1]} --cfg ./config/model_config.yaml \
                --weights {self.weights} --name {self.model_name} --hyp ./config/hyp.scratch.custom.yaml --epochs {self.num_epochs} --exist-ok")

            shutil.move(f"./runs/train/{self.model_name}/weights/best.pt", self.trained_weights) # move trained weights to checkpoint folder
            print("done training")

            self.weights = self.trained_weights # updating to trained weights
            
            return {
                'model_path': self.trained_weights,
            }
        else:
            print('No labeled tasks found: make some annotations...')
            return {}
    
    def predict(self, tasks, **kwargs):
        print("start predictions")
        results = []
        all_scores= []
        
        for task in tasks:
           
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url, project_dir=self.image_dir)
            img = Image.open(image_path)
            img_width, img_height = get_image_size(image_path)
            
            preds = self.model(img, size=img_width)
            preds_df = preds.pandas().xyxy[0]
            
            for x_min,y_min,x_max,y_max,confidence,class_,name_ in zip(preds_df['xmin'],preds_df['ymin'],
                                                                        preds_df['xmax'],preds_df['ymax'],
                                                                        preds_df['confidence'],preds_df['class'],
                                                                        preds_df['name']):
                # add label name from label_map
                output_label = self.label_map.get(name_, name_)
                if output_label not in self.labels_in_config:
                    continue
                
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    "original_width": img_width,
                    "original_height": img_height,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [name_],
                        'x': x_min / img_width * 100,
                        'y': y_min / img_height * 100,
                        'width': (x_max - x_min) / img_width * 100,
                        'height': (y_max - y_min) / img_height * 100
                    },
                    'score': confidence
                })
                all_scores.append(confidence)

        avg_score = sum(all_scores) / max(len(all_scores), 1)
        
        return [{
            'result': results,
            'score': avg_score
        }]
