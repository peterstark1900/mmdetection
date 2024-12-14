'''
@brief The code of cropping the image and transforming the keypoints
@author Peter Stark
@date 2021-12-10
@version v0.2
'''
import math
import json
import cv2
import json
import os

''' Module of generating customized dataset for mmpose from CVAT

This module is based on the COCO keypoints-1.0 format in CVAT. By using this module, the user can crop the image and transform the keypoints in the annotation file. The cropped image and the transformed keypoints will be saved in the new dataset. The new dataset will be saved in the specified path. The width and height of the cropped image can be set by the user. The user can also specify the path of the original dataset and the new dataset.
'''



def calculate_bbox(center_piont, w, h):
    '''function to calculate the bounding box of the cropped image
    Args:
        center_piont: the center point of the object
        w: the width of the cropped image
        h: the height of the cropped image
    Returns:
        x1, y1, x2, y2: the coordinates of the bounding box(x1, y1, x2, y2), which are the top-left and bottom-right points of the bounding box. This is different from (x, y, w, h)!!!
    '''
    x, y = center_piont
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return x1, y1, x2, y2


def cut_with_bbox(raw_image,x1, y1, x2, y2,file_name):
    '''function to crop the image with the bounding box
    Args:
        raw_image: the original image
        x1, y1, x2, y2: the coordinates of the bounding box(x1, y1, x2, y2), which are the top-left and bottom-right points of the bounding box. This is different from (x, y, w, h)!!!
        file_name: the name of the image file
    Returns:
        cropped_image: the cropped image
        cropped_flag: a flag to indicate whether the cropping is successful. If the cropping is successful, the flag is True. Otherwise, the flag is False.
    '''
    height, width, _ = raw_image.shape
    if x1< 0 or y1 < 0 or x2 > width or y2 > height:
        print(f"Invalid crop area for image: {file_name}")
        print(f"Image size: {width}x{height}")
        print(f"Crop area: {x1}, {y1}, {x2}, {y2}")
        cropped_flag =  False
        return None, cropped_flag
    else:
        cropped_image = raw_image[int(y1):int(y2), int(x1):int(x2)]
        cropped_flag = True
        return cropped_image, cropped_flag
    
def keypoints_tranformation(keypoints, x1, y1):
    ''' function to transform the keypoints
    Args:
        keypoints: the keypoints of the object
        x1, y1: the top-left point of the bounding box
    Returns:
        new_keypoints: the transformed keypoints
    '''
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

class CustomizeDataset:
    ''' 
    Class of generating customized dataset for mmpose from CVAT
    '''
    def __init__(self,image_output_path,json_output_path,w,h):
        '''function to initialize the dataset
        Args:
            image_output_path: the path to save the cropped images
            json_output_path: the path to save the new dataset
            w: the width of the cropped image
            h: the height of the cropped image
        '''
        self.images_list = []
        self.annotations_list = []
        self.categories_list = []
        self.dataset_categories = []
        self.image_name_counter = 0
        self.image_save_path = image_output_path
        self.json_export_path = json_output_path
        self.width = w
        self.height = h
        #clear the content in the "json_output_path" and "image_output_path":
        with open(json_output_path, 'w') as f:
            f.write('')
        for file in os.listdir(image_output_path):
            os.remove(image_output_path + file)

    
    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def update_list(self, values, list_type):
        '''function to update the list of categories, images, and annotations
        Args:
            values: the values to be updated
            list_type: the type of the list
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
        '''function to match and return the category id
        Args:
            category_name: the name of the category
        Returns:    
            the category id
        '''
        for i in range(len(self.categories_list)):
            if self.categories_list[i]['name'] == category_name:
                return self.categories_list[i]['id']

    def allocate_image_id(self):
        self.image_name_counter += 1
        
    def get_image_id(self):
        return self.image_name_counter

    def save_image(self, image):
        '''function to save the image
        Args:   
            image: the image to be saved
        '''
        image_path = self.image_save_path + str(self.image_name_counter) + '.PNG'
        cv2.imwrite(image_path, image)
    
    def export_json(self):
        '''
        function to export the dataset to a json file
        '''
        data = {
            "categories": self.categories_list,
            "images": self.images_list,
            "annotations": self.annotations_list
            
        }
        with open(self.json_export_path, 'w') as f:
            json.dump(data, f, indent=4)


def file_pipeline(json_input_path,image_input_path,my_dataset):
    '''function to process the dataset
    Args:
        json_input_path: the path of the original json file
        image_input_path: the path of the original images
        my_dataset: the object of the dataset
    '''

    w = my_dataset.get_width()
    h = my_dataset.get_height()

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
        #select keypoint "body" as the center point
        x = annotation['keypoints'][3]
        y = annotation['keypoints'][4]
        center_point = (x, y)
        x1, y1, x2, y2 = calculate_bbox(center_point, w, h)
        # find the file name of the image
        image_id = annotation['image_id']
        file_name = image_id_to_file_name.get(image_id, None)
        # read the original image
        image_file_path = image_input_path+f'{file_name}'
        raw_image = cv2.imread(image_file_path)
        if raw_image is None:
            print(f"Image not found: {image_file_path}")
            continue

        # crop the image
        cropped_image,cropped_flag = cut_with_bbox(raw_image, x1, y1, x2, y2,file_name)
        if cropped_flag == False:
            print(f"Fail to crop the {file_name}, skip this image")
            continue
        else:
            print(f"Successfully crop the {file_name}")
            # save the cropped image
            my_dataset.allocate_image_id()
            my_dataset.save_image(cropped_image)
            # get the information of the original image in `data['images']` 
            temp_image_value = [image for image in data['images'] if image['id'] == image_id]
            #update the image information
            temp_image_value[0]['id'] = my_dataset.get_image_id()
            temp_image_value[0]['file_name'] = f"{my_dataset.get_image_id()}.PNG"
            temp_image_value[0]['width'] = w
            temp_image_value[0]['height'] = h
            my_dataset.update_list(temp_image_value[0], 'images')
        
        # modify the annotation information
        new_keypoints = keypoints_tranformation(annotation['keypoints'], x1, y1)
        annotation['bbox'] = 0,0,w,h
        annotation['keypoints'] = new_keypoints
        annotation['area'] = w*h
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





def sample_check_for_cropping(json_input_path,image_input_path,w,h,sample_id):
    '''function to check the cropping result
    Args:
        json_input_path: the path of the original json file
        image_input_path: the path of the original images
        w: the width of the cropped image
        h: the height of the cropped image
        sample_id: the id of the sample to be checked
    '''
    with open(json_input_path, 'r') as f:
        data = json.load(f)
    
    # create a dictionary to store the image id and file name
    image_id_to_file_name = {image['id']: image['file_name'] for image in data['images']}

    annotation = data['annotations'][sample_id]
    print(annotation['keypoints'])
    #select keypoint "body" as the center point
    x = annotation['keypoints'][3]
    y = annotation['keypoints'][4]
    # print(x, y)
    center_point = (x, y)
    x1, y1, x2, y2 = calculate_bbox(center_point, w,h)
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
        
        # crop the image
        cropped_image,cropped_flag = cut_with_bbox(raw_image, x1, y1, x2, y2,file_name)
        # transform the keypoints
        new_keypoints = keypoints_tranformation(annotation['keypoints'], x1, y1)
        # draw the keypoints on the cropped image
        for i in range(0, len(new_keypoints), 3):
            x = new_keypoints[i]
            y = new_keypoints[i+1]
            v = new_keypoints[i+2]
            if v == 2:
                cv2.circle(cropped_image, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.imshow('result.png', cropped_image)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    



def main():
    '''function to run the pipeline

    The main function could be devided into 3 parts:
    1. Setup the output path of the new dataset
    2. Setup the input path for the training dataset
    3. Setup the input path for the testing dataset

    Before running the main function, the user could use the fuction "sample_check_for_cropping" to check the cropping result. 
    '''
##############################################################
    # setup the training and testing dataset
    train_json_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/annotations/Fish-Tracker-1210-Train.json'
    train_image_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/images/Train/'
    fish1210_dataset_train = CustomizeDataset(train_image_output_path, train_json_output_path, 256, 256)

    test_json_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/annotations/Fish-Tracker-1210-Test.json'
    test_image_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/images/Test/'
    fish1210_dataset_test = CustomizeDataset(test_image_output_path, test_json_output_path, 256, 256)
##############################################################

    # train_json_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/VID_20241210_155634/annotations/person_keypoints_Train.json'
    # train_json_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/annotations/Fish-Tracker-1210-Train.json'
    # train_image_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/VID_20241210_155634/images/Train/'
    # train_image_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/images/Train/'

    demo4_image_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/fish-1210-demo4/images/Train/'
    demo4_json_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/fish-1210-demo4/annotations/fish-1210-demo4.json'
    # sample_check_for_cropping(demo4_json_input_path, demo4_image_input_path, w=256, h=256, sample_id = 12)
    file_pipeline(demo4_json_input_path,demo4_image_input_path,fish1210_dataset_train)
##############################################################

    # test_json_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/VID_20241210_160115/annotations/person_keypoints_Test.json'
    # test_json_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/annotations/Fish-Tracker-1210-Test.json'
    # test_image_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/VID_20241210_160115/images/Test/'
    # test_image_output_path = '/home/peter/mmpose/data/Fish-Tracker-1210/images/Test/'

    demo1_json_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/fish-1210-demo1/annotations/fish-1210-demo1.json'
    demo1_image_input_path = '/home/peter/Desktop/Fish-Dataset/fish-1210/fish-1210-demo1/images/Test/'
    # sample_check_for_cropping(demo1_json_input_path, demo1_image_input_path, w=256, h=256, sample_id = 256)
    file_pipeline(demo1_json_input_path,demo1_image_input_path,fish1210_dataset_test)
##############################################################

if __name__ == '__main__':
    main()
    