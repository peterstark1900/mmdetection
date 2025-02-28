import math
import json
import cv2
import json
import os
import shutil
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
from tools.peter import PeterDataset,formatting_json

def pipeline_export_total():
    # create the dataset
    my_total_output_path = '/home/peter/Desktop/Fish-Dataset/Fish-0223/2CVAT/TEST-MAM30B15W15R15E'
    my_json_name = 'TEST-MAM30B15W15R15E'
    dataset = PeterDataset(save_flag = True, export_type= 'total',json_name=my_json_name,total_output_path=my_total_output_path,split_output_path = None)
    # setup the categories
    dataset.setup_categories(1,"myfish"," ",["head","body","joint","tail"],[[3,2],[1,2],[3,4]])

    # load the images and annotations from the temp source
    json_source_path_1 = '/home/peter/Desktop/Fish-Dataset/Fish-0223/model_outcome/TEST-MAM30B15W15R15E/annotations'
    image_source_path_1 = '/home/peter/Desktop/Fish-Dataset/Fish-0223/extract_outcome/TEST-MAM30B15W15R15E'
    dataset.load_from_temp_source(json_source_path_1,image_source_path_1,'customize')

    # finish the loading task
    dataset.finish_loding()

    # export the dataset
    dataset.export_original()

def pipeline_export_corpped():
    # create the dataset
    my_split_output_path = '/home/peter/mmpose/data/Fish-Tracker-1222'
    my_json_name = 'Fish-Tracker-1222'
    dataset = PeterDataset(save_flag = True, export_type= 'split',json_name=my_json_name,total_output_path=None, split_output_path = my_split_output_path)

    # load the images and annotations from the temp source
    json_source_path_1 = '/home/peter/Desktop/Fish-Dataset/Fish-0223/real_annotations/fish-0223-demo1.json'
    image_source_path_1 = '/home/peter/Desktop/Fish-Dataset/Fish-0223/extract_outcome/fish-0223-demo1'
    dataset.load_from_temp_source(json_source_path_1,image_source_path_1,'cvat')

    # load the images and annotations from the temp source
    json_source_path_2 = '/home/peter/Desktop/Fish-Dataset/Fish-0223/real_annotations/TEST-MAM30B15W15R15E.json'
    image_source_path_2 = '/home/peter/Desktop/Fish-Dataset/Fish-0223/extract_outcome/TEST-MAM30B15W15R15E'
    dataset.load_from_temp_source(json_source_path_2,image_source_path_2,'cvat') 

    # load the images and annotations from the temp source
    json_source_path_3 = '/home/peter/Desktop/Fish-Dataset/fish-1222/fish-1222-demo18/annotations/fish-1222-demo18.json'
    image_source_path_3 = '/home/peter/Desktop/Fish-Dataset/fish-1222/fish-1222-demo18/images/Train/'
    dataset.load_from_temp_source(json_source_path_3,image_source_path_3,'cvat')

    # load the images and annotations from the temp source
    json_source_path_4 = '/home/peter/Desktop/Fish-Dataset/fish-1222/fish-1222-demo19/annotations/fish-1222-demo19.json'
    image_source_path_4 = '/home/peter/Desktop/Fish-Dataset/fish-1222/fish-1222-demo19/images/Test/'
    dataset.load_from_temp_source(json_source_path_4,image_source_path_4,'cvat')


    # finish the loading task
    dataset.finish_loding()

    # split the dataset
    dataset.split_dataset(0.8)

    # export the dataset
    dataset.export_cropped(256,256)

def pipeline_check_cropped():

    # create the dataset
    dataset = PeterDataset(save_flag = False)

    # load the images and annotations from the temp source
    json_source_path_1 = '/home/peter/Desktop/Fish-Dataset/Fish-0223/real_annotations/fish-0223-demo1.json'
    image_source_path_1 = '/home/peter/Desktop/Fish-Dataset/Fish-0223/extract_outcome/fish-0223-demo1'
    dataset.load_from_temp_source(json_source_path_1,image_source_path_1,'cvat')

    # load the images and annotations from the temp source
    json_source_path_2 = '/home/peter/Desktop/Fish-Dataset/Fish-0223/real_annotations/TEST-MAM30B15W15R15E.json'
    image_source_path_2 = '/home/peter/Desktop/Fish-Dataset/Fish-0223/extract_outcome/TEST-MAM30B15W15R15E'
    dataset.load_from_temp_source(json_source_path_2,image_source_path_2,'cvat') 

    # load the images and annotations from the temp source
    json_source_path_3 = '/home/peter/Desktop/Fish-Dataset/fish-1222/fish-1222-demo18/annotations/fish-1222-demo18.json'
    image_source_path_3 = '/home/peter/Desktop/Fish-Dataset/fish-1222/fish-1222-demo18/images/Train/'
    dataset.load_from_temp_source(json_source_path_3,image_source_path_3,'cvat')

    # load the images and annotations from the temp source
    json_source_path_4 = '/home/peter/Desktop/Fish-Dataset/fish-1222/fish-1222-demo19/annotations/fish-1222-demo19.json'
    image_source_path_4 = '/home/peter/Desktop/Fish-Dataset/fish-1222/fish-1222-demo19/images/Test/'
    dataset.load_from_temp_source(json_source_path_4,image_source_path_4,'cvat')


    # finish the loading task
    dataset.finish_loding()

    # check the cropping outcome directly
    dataset.try_cropped(256,256,999)

def main():

    # formatting_json('/home/peter/Desktop/Fish-Dataset/fish-1222/fish-1222-demo18/annotations/fish-1222-demo18.json')

    # # pipeline_export_total()
    pipeline_export_corpped()
    # pipeline_check_cropped()

if __name__ == '__main__':
    main()