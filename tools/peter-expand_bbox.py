'''
@brief This script is used to expand the bounding box 
@author Peter Stark
@date 2024-12-13
@version v1.1
'''
import math
import json
import cv2
import json
import os
''' Module for expanding the bounding box

The origianl dataset from CVAT is not suitable for mmdetection because the bbox so tight that the fish is merely in the bbox, which would affect the performance of the mmpose model. Therefore, module are used to expand the bbox to make sure the fish is in the bbox.
'''
#expand the bbox with scale
def expand_bbox(raw_bbox, scale):
    ''' function to expand the bounding box with a scale
    Args:
        `raw_bbox`: list, [x1, y1, w, h]
        `scale`: float, the scale to expand the bbox
    Returns:
        `new_x1`, `new_y1`, `new_w`, `new_h`: float, the new bounding box
    '''
    x1, y1, w, h = raw_bbox
    x2 = x1 + w
    y2 = y1 + h
    # calculate the center of the bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    # calculate the new width and height
    new_w = w * scale
    new_h = h * scale
    # calculate the new x1, y1, x2, y2
    new_x1 = center_x - new_w / 2
    new_y1 = center_y - new_h / 2
    return new_x1, new_y1, new_w, new_h
   

class CustomizeDataset:
    def __init__(self,image_output_path,json_output_path):
        ''' function to initialize the dataset
        Args:
            `image_output_path`: str, the path to save the images
            `json_output_path`: str, the path to save the json file
        '''
        self.images_list = []
        self.annotations_list = []
        self.categories_list = []
        self.dataset_categories = []
        self.image_name_counter = 0
        self.image_save_path = image_output_path
        self.json_export_path = json_output_path
        #clear the content in the "json_output_path" and "image_output_path":
        with open(json_output_path, 'w') as f:
            f.write('')
        for file in os.listdir(image_output_path):
            os.remove(image_output_path + file)
    
    def update_list(self, values, list_type):
        ''' function to update the list of categories, images, and annotations
        Args:
            `values`: list, the values to be added to the list
            `list_type`: str, the type of the list
        '''
        if list_type == 'categories':
            for value in values:
                if value['name'] not in self.dataset_categories:
                    value['id'] = len(self.dataset_categories) + 1
                    self.dataset_categories.append(value['name'])
                    self.categories_list.append(value)
                else:
                    print(f"Category {value['name']} already exists in the dataset")
        elif list_type == 'images':
                self.images_list.append(values)
        elif list_type == 'annotations':
                self.annotations_list.append(values)

    # match and return the category id 
    def get_catergory_id(self, category_name):
        for i in range(len(self.categories_list)):
            if self.categories_list[i]['name'] == category_name:
                return self.categories_list[i]['id']

    def allocate_image_id(self):
        self.image_name_counter += 1
        
    def get_image_id(self):
        return self.image_name_counter

    def save_image(self, image):
        image_path = self.image_save_path + str(self.image_name_counter) + '.PNG'
        cv2.imwrite(image_path, image)
    
    def export_json(self):
        data = {
            "categories": self.categories_list,
            "images": self.images_list,
            "annotations": self.annotations_list
            
        }
        with open(self.json_export_path, 'w') as f:
            json.dump(data, f, indent=4)


def file_pipeline(json_input_path,image_input_path,my_dataset,scale):
    ''' function to process the json file and images
    Args:
        `json_input_path`: str, the path to the json file
        `image_input_path`: str, the path to the images
        `my_dataset`: object, the dataset object
        `scale`: float, the scale to expand the bounding box
    '''
    # open the original json file
    with open(json_input_path, 'r') as f:
        data = json.load(f)

    # create a dictionary to store the image id and file name
    image_id_to_file_name = {image['id']: image['file_name'] for image in data['images']}

    # create a dictionary to store the category id and category name
    category_id_to_name = {category['id']: category['name'] for category in data['categories']}

    # update the categories list in the dataset
    my_dataset.update_list(data['categories'], 'categories')

    # iterate through all the annotations
    for i in range(len(data['annotations'])):
        annotation = data['annotations'][i]
        new_x1, new_y1, new_w, new_h = expand_bbox(annotation['bbox'], scale)
        # find the file name of the image
        image_id = annotation['image_id']
        file_name = image_id_to_file_name.get(image_id, None)
        # read the original image
        image_file_path = image_input_path+f'{file_name}'
        raw_image = cv2.imread(image_file_path)
        if raw_image is None:
            print(f"Image not found: {image_file_path}")
            continue

        print(f"Successfully laod the {file_name}")
        # save the cropped image
        my_dataset.allocate_image_id()
        my_dataset.save_image(raw_image)
        # get the information of the original image in `data['images']` 
        temp_image_value = [image for image in data['images'] if image['id'] == image_id]
        #update the image information
        temp_image_value[0]['id'] = my_dataset.get_image_id()
        temp_image_value[0]['file_name'] = f"{my_dataset.get_image_id()}.PNG"
        my_dataset.update_list(temp_image_value[0], 'images')
        
        # modify the annotation information
        annotation['bbox'] = new_x1, new_y1, new_w, new_h
        annotation['area'] = new_w*new_h
        annotation['id'] = my_dataset.get_image_id()
        annotation['image_id'] = my_dataset.get_image_id()
        # get the category name
        category_name = category_id_to_name.get(annotation['category_id'], None)
        # update category id by the category name above
        annotation['category_id'] = my_dataset.get_catergory_id(category_name)
        # update the annotation information
        my_dataset.update_list(annotation, 'annotations')
    # export the new json file
    my_dataset.export_json()

    print(f'Dataset has been successfully created and saved to {my_dataset.json_export_path}')
    print(f'Total number of images: {my_dataset.get_image_id()}')
    print(' ')





def sample_check_for_expanding(json_input_path,image_input_path,scale,sample_id):
    ''' function to check the result of expanding the bounding box
    Args:
        `json_input_path`: str, the path to the json file
        `image_input_path`: str, the path to the images
        `scale`: float, the scale to expand the bounding box
        `sample_id`: int, the index of the sample to be checked
    '''
    
    with open(json_input_path, 'r') as f:
        data = json.load(f)
    
    # create a dictionary to store the image id and file name
    image_id_to_file_name = {image['id']: image['file_name'] for image in data['images']}

    annotation = data['annotations'][sample_id]
    print(annotation['bbox'])
    new_x1, new_y1, new_w, new_h = expand_bbox(annotation['bbox'], scale)
    # find the file name of the image
    image_id = annotation['image_id']
    file_name = image_id_to_file_name.get(image_id, None)
    # read the original image
    image_file_path = image_input_path+f'{file_name}'
    raw_image = cv2.imread(image_file_path)
    if raw_image is None:
        print(f"Image not found: {image_file_path}")
        return
    else:
        print(f"Successfully load the {file_name}")
        
        keypoints = annotation['keypoints']
        # draw the keypoints and the new bbox on the raw image:
        cv2.rectangle(raw_image, (int(new_x1), int(new_y1)), (int(new_x1+new_w), int(new_y1+new_h)), (0, 255, 0), 2)
        for i in range(0, len(keypoints), 3):
            x = keypoints[i]
            y = keypoints[i+1]
            v = keypoints[i+2]
            if v == 2:
                cv2.circle(raw_image, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.imshow('result.png', raw_image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    



def main():
    ''' function to run the the pipeline

    The main fucntion could be devided into 3 parts:
    1. Setup the output path of the new dataset
    2. Setup the input path for the training dataset
    3. Setup the input path for the testing dataset

    Before running the main function, the user coudl use the function `sample_check_for_expanding` to check the result of expanding the bounding box    
    '''
##############################################################
    # setup the training and testing dataset
    train_json_output_path = '/home/peter/mmdetection/data/Fish-Tracker-1210/annotations/Fish-Tracker-1210-Train.json'
    train_image_output_path = '/home/peter/mmdetection/data/Fish-Tracker-1210/images/Train/'
    fish1210_dataset_train = CustomizeDataset(train_image_output_path, train_json_output_path)

    test_json_output_path = '/home/peter/mmdetection/data/Fish-Tracker-1210/annotations/Fish-Tracker-1210-Test.json'
    test_image_output_path = '/home/peter/mmdetection/data/Fish-Tracker-1210/images/Test/'
    fish1210_dataset_test = CustomizeDataset(test_image_output_path, test_json_output_path)
##############################################################

    # train_json_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/VID_20241210_155634/annotations/person_keypoints_Train.json'
    # train_json_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/annotations/Fish-Tracker-1210-Train.json'
    # train_image_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/VID_20241210_155634/images/Train/'
    # train_image_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/images/Train/'

    demo4_image_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/fish-1210-demo4/images/Train/'
    demo4_json_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/fish-1210-demo4/annotations/fish-1210-demo4.json'
    # sample_check_for_expanding(demo4_json_input_path, demo4_image_input_path, scale = 2.0, sample_id = 300)
    file_pipeline(demo4_json_input_path,demo4_image_input_path,fish1210_dataset_train,scale = 1.5)
##############################################################

    # test_json_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/VID_20241210_160115/annotations/person_keypoints_Test.json'
    # test_json_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/annotations/Fish-Tracker-1210-Test.json'
    # test_image_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/VID_20241210_160115/images/Test/'
    # test_image_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/images/Test/'

    demo1_json_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/fish-1210-demo1/annotations/fish-1210-demo1.json'
    demo1_image_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/fish-1210-demo1/images/Test/'
    # sample_check_for_expanding(demo1_json_input_path, demo1_image_input_path, scale = 2, sample_id = 123)
    file_pipeline(demo1_json_input_path,demo1_image_input_path,fish1210_dataset_test,scale = 1.5)
##############################################################

if __name__ == '__main__':
    main()
    