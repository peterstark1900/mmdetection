import json
import os
import random
import shutil

def read_multiple_json(source_dir):
    all_json_files = [f for f in os.listdir(source_dir) 
                   if f.lower().endswith(('json'))]
    print(len(all_json_files))

def read_single_json(json_input_path):
    with open(json_input_path, 'r') as f:
        data = json.load(f)
    # keypoints = data[0]['keypoints']
    # print(keypoints)
    # # print(keypoints[0][0])
    # # print(keypoints[0][1])
    # # print(keypoints[1][0])
    # bbox = data[0]['bbox'][0]
    # print(bbox)
    # test_unit = {"image_id": 0, "category_id": 1, "keypoints": [keypoints[0][0],keypoints[0][1],2,keypoints[1][0],keypoints[1][1],2,keypoints[2][0],keypoints[2][1],2,keypoints[3][0],keypoints[3][1],2], "num_keypoints": 4, "bbox": bbox, "area": 0, "iscrowd": 0}
    # print(test_unit)

    print(len(data['annotations']))
    print(len(data['images']))

def suffle_test():
    temp_list = [[1,2],[3,4],[5,6],[7,8],[9,10]]
    print(temp_list)
    random.shuffle(temp_list)
    print(temp_list)

def main():
    json_input_path = '/home/peter/mmpose/data/Fish-Tracker-1222/annotations/Test/Fish-Tracker-1222-Test.json'
    read_single_json(json_input_path)
    # suffle_test()



if __name__ == '__main__':
    main()