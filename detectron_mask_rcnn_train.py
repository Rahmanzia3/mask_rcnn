import cv2
import wget
import requests
import detectron2
import numpy as np
from tqdm import tqdm
import torch, torchvision
from zipfile import ZipFile 
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from argparse import ArgumentParser
import argparse

def select_files(path):


    sub_foders = os.listdir(path)
    train = []
    test = []
    names = []
    for x in sub_foders:
        if x == 'test':
            test_dir = os.path.join(path,x)
            test.append(test_dir)
        elif x == 'train':
            train_dir = os.path.join(path,x)
            train.append(train_dir)
        elif x == 'names.txt':
            names_dir = os.path.join(path,x)
            names.append(names_dir)

    return train[0], test[0], names[0]

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as bar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        bar.update(CHUNK_SIZE)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination) 

def un_zip(source,destination):
        file_name = source
      
        # opening the zip file in READ mode 
        with ZipFile(file_name, 'r') as zip: 
        # printing all the contents of the zip file 
            zip.printdir() 
          
            # extracting all the files 
            print('Extracting all the files now...') 

            # ######## ADD DESTINATION LOCATION HERE
            zip.extractall(destination) 
            print('Done!') 

if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument("--backbone", default='mask_rcnn', help="Defauls is mask_RCNN")
    parser.add_argument("--project", required=True, help="Give project name")
    parser.add_argument("--batch_size", default = 3 , help="Give batch size")
    parser.add_argument("--steps", default =  250, help="Give batch step size")
    parser.add_argument("--epochs", default =  25, help="Give epoch size or save weights at", type = int)
    parser.add_argument("--data_source", required = True, help='Provide url or google Id or path(dir)')

    opt = parser.parse_args()

    batch_size_get = opt.batch_size
    epochs_get = opt.epochs
    steps_get = opt.steps


    # making project folder
    project_name = opt.project 
    project_dir = os.path.join(os.getcwd(),project_name)

    isdir = os.path.isdir(project_dir) 
    if isdir is False:
        # CREATING PROJECT DIRECTORY
        os.mkdir(project_dir)



    input_data_source = opt.data_source

    is_dir = os.path.isdir(input_data_source)


    train_path = []
    test_path = []
    names_path = []
    if is_dir is True:
        print('input_data_source is a directory')
        train_directory , test_directory, names_directory = select_files(input_data_source)

        # print('Train',train_directory)
        # print('Test',test_directory)
        # print('Names',names_directory)

    elif is_dir is False:
        '''
            aFTER DOWNLOADING DATA FROM SOURCE UNZIP IT AND THEN PASS THE DATA DIR HERE TO FIND TEST TRAIN AND NAMES

            DATA SHOULD BE DOWNLOADED IN PROJECT DIRECTORY

        '''

        is_url = input_data_source.find('https')

        if is_url != -1:
            print('input_data_source is a url')
            # Download data in project directory
            wget.download(input_data_source,project_dir)
            sub_dir = os.listdir(project_dir)
            for x in sub_dir:
                is_zip = x.split('.')[-1]
                if is_zip == 'zip':
                    zip_path = os.path.join(project_dir,x)
            # zip file path
            print(zip_path)
            # source zip / project directory
            un_zip(zip_path,project_dir)

        elif is_url == -1 and is_dir == False:
            print('input_data_source is a google id')
            print('KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK')
            # while downloading frm google drive we need to provide .zip path
            download_path = os.path.join(project_dir,'baloon.zip')
            download_file_from_google_drive(input_data_source,download_path)
            sub_dir = os.listdir(project_dir)
            un_zip(download_path,project_dir)


        sub_project_dir = os.listdir(project_dir)
        down_fodler = []
        for x in sub_project_dir:
            folder_path = os.path.join(project_dir,x)

            is_downloaded_folder = os.path.isdir(folder_path)
            if is_downloaded_folder is True:
                downld_folder_is = is_downloaded_folder
                down_fodler.append(folder_path)
            else:
                pass

        input_data_source = down_fodler[0]
        print('$$$$$$$$$$$$$$$$$$$$$444')
        print(input_data_source)
        print('$$$$$$$$$$$$$$$$$$$$$444')

        train_directory , test_directory, names_directory = select_files(input_data_source)




    def get_balloon_dicts(img_dir):
        json_file = os.path.join(img_dir, "via_region_data.json")
        with open(json_file) as f:
            imgs_anns = json.load(f)

        dataset_dicts = []
        for idx, v in enumerate(imgs_anns.values()):
            record = {}
            
            filename = os.path.join(img_dir, v["filename"])
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
          
            annos = v["regions"]
            objs = []
            for _, anno in annos.items():
                assert not anno["region_attributes"]
                anno = anno["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts

    for d in ["train", "test"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts(input_data_source+"/" + d))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
    balloon_metadata = MetadataCatalog.get("balloon_train")

    """To verify the data loading is correct, let's visualize the annotations of randomly selected samples in the training set:"""

    dataset_dicts = get_balloon_dicts(train_directory)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("Verify_data",out.get_image()[:, :, ::-1])
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    """## Train!

    Now, let's fine-tune a COCO-pretrained R50-FPN Mask R-CNN model on the balloon dataset. It takes ~6 minutes to train 300 iterations on Colab's K80 GPU, or ~2 minutes on a P100 GPU.
    """
    # Get number of classes here[!]


    file_read_names = open(names_directory)
    read_lines = file_read_names.readlines() 
    count = 0
    for x in read_lines:
        if len(x) != 1:
            count = count + 1

    number_classes = count

    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.CHECKPOINT_PERIOD = int(epochs_get)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = int(batch_size_get)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.MAX_ITER = int(steps_get)    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = number_classes  # only has one class (ballon)
    cfg.OUTPUT_DIR = project_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()




# google dirive downloadeble link
    
# 17OCk2sCV1rngxL1_-MjpB0gypF85fIKU