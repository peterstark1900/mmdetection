import math
import json
import cv2
import json

def calculate_bbox(center_piont, w, h):
    x, y = center_piont
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return x1, y1, x2, y2

def cut_with_bbox(raw_image,x1, y1, x2, y2,file_name):
    height, width, _ = raw_image.shape
    # print(image_file_path)
    # print(raw_image.shape)
    # print(' ')
    print(x1, y1, x2, y2)
    print(width, height)
    if x1< 0 or y1 < 0 or x2 > width or y2 > height:
        print(f"Invalid crop area for image: {file_name}")
    else:
        cropped_image = raw_image[int(y1):int(y2), int(x1):int(x2)]
        return cropped_image
    
def keypoints_tranformation(keypoints, x1, y1):
    new_keypoints = []
    for i in range(0, len(keypoints), 3):
        x = keypoints[i]
        y = keypoints[i+1]
        v = keypoints[i+2]
        if v == 0:
            new_keypoints.extend([0, 0, 0])
        else:
            new_keypoints.extend([x-x1, y-y1, 2])
    return new_keypoints

def file_pipeline(json_input_path, json_output_path,image_input_path,image_output_path):
    with open(json_input_path, 'r') as f:
        data = json.load(f)
    for i in range(len(data['annotations'])):
        annotation = data['annotations'][i]
        bbox = annotation['bbox']
        x, y, width, height = bbox
        center_point = (x+width/2, y+height/2)
        new_bbox = calculate_bbox(center_point, 100, 200)
        x1, y1, x2, y2 = new_bbox



        image_id_to_file_name = {image['id']: image['file_name'] for image in data['images']}
        image_id = annotation['image_id']
        file_name = image_id_to_file_name.get(image_id, None)
        image_file_path = image_input_path+f'{file_name}'
        # 读取图片
        old_image = cv2.imread(image_file_path)
        if old_image is None:
            print(f"Image not found: {image_file_path}")
            continue
        image_file_path = image_input_path+f'{file_name}'

        raw_image = cv2.imread(image_file_path)
        cropped_image = cut_with_bbox(raw_image, x1, y1, x2, y2,)
        new_keypoints = keypoints_tranformation(annotation['keypoints'], x1, y1)
        annotation['bbox'] = new_bbox
        annotation['keypoints'] = new_keypoints
        cv2.imwrite(image_output_path + + f'{file_name}', cropped_image)
    # 将修改后的数据写回到 JSON 文件中
    with open(json_output_path, 'w') as f:
        json.dump(data, f, indent=4)

def sample_test_for_bbox(json_input_path,image_input_path):
    with open(json_input_path, 'r') as f:
        data = json.load(f)
    annotations = data['annotations'][0]
    print(annotations['keypoints'])

    # print(annotations)
    # bbox = annotations['bbox']
    # x, y, width, height = bbox
    # center_point = (x+width/2, y+height/2)
    x = annotations['keypoints'][3]
    y = annotations['keypoints'][4]
    # print(x, y)
    center_point = (x, y)
    new_bbox = calculate_bbox(center_point, 840, 840)
    x1, y1, x2, y2 = new_bbox
    image_id_to_file_name = {image['id']: image['file_name'] for image in data['images']}
    image_id = annotations['image_id']
    file_name = image_id_to_file_name.get(image_id, None)
    image_file_path = image_input_path+f'{file_name}'
    raw_image = cv2.imread(image_file_path)
    if raw_image is None:
        print(f"Image not found: {image_file_path}")
        return
    cropped_image = cut_with_bbox(raw_image, x1, y1, x2, y2,file_name)
    cv2.imshow('result.png', cropped_image)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()



def main():


    train_json_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/VID_20241210_155634/annotations/person_keypoints_Train.json'
    train_json_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/annotations/Fish-Tracker-1210-Train.json'
    train_image_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/VID_20241210_155634/images/Train/'
    train_image_output_path = '/home/peter/mmpose/data/Fish-Tracker-1202/images/Train/'

    sample_test_for_bbox(train_json_input_path, train_image_input_path)
    # file_pipeline(train_json_input_path, train_json_output_path,train_image_input_path,train_image_output_path)

    # test_json_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/VID_20241210_160115/annotations/person_keypoints_Test.json'
    # test_json_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/annotations/Fish-Tracker-1210-Test.json'
    # test_image_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/VID_20241210_160115/images/Test/'
    # test_image_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/images/Test/'
    # file_pipeline(test_json_input_path, test_json_output_path,test_image_input_path,test_image_output_path)

if __name__ == '__main__':
    main()
    