import math
import json
import cv2
import json
import os
import shutil
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))
from tools.peter import PeterDataset

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


def main():

    # formatting_json('/home/peter/Desktop/Fish-Dataset/Fish-0223/real_annotations/TEST-MAM30B15W15R15E.json')

    pipeline_export_total()
    # pipeline_export_corpped()
    # pipeline_check_cropped()

if __name__ == '__main__':
    main()