import math
import json
import cv2
import json
import os
import shutil
import numpy as np
import tqdm


class PeterDataset:
    ''' 
    Class of generating customized dataset for mmpose 
    '''
    def __init__(self,save_flag,export_type = None,json_name = None,total_output_path = None,split_output_path = None):
        '''Function to initialize the dataset
        Args:
            `image_output_path`: the path to save the processed images
            `json_output_path`: the path to save the new dataset
        '''
        self.save_flag = save_flag
        self.name_of_categories = []
        self.dataset_categories_list = []
        self.counter = 0
        self.total_list = []
        self.image_source_paths = []
        self.json_source_paths = []
        self.image_dict = {}
        if self.save_flag == True:
            self.json_name = json_name
            self.export_type = export_type
            print(f'export_type: {self.export_type}')
            if self.export_type == 'total':
                # create the path to save the images and the file of json format
                self.total_image_save_path = os.path.join(total_output_path, 'images')
                self.total_json_export_path = os.path.join(total_output_path, 'annotations',f'{self.json_name}_total.json')
                print(f'total_image_save_path: {self.total_image_save_path}')
                print(f'total_json_export_path: {self.total_json_export_path}')
                # print(f'total_image_save_path: {self.total_image_save_path}')
                # print(f'total_json_export_path: {self.total_json_export_path}')
                # if the path does not exist, create the path
                if not os.path.exists(os.path.dirname(self.total_image_save_path)):
                    os.makedirs(os.path.dirname(self.total_image_save_path))
                    print(f"Directory {self.total_image_save_path} created.")
                if not os.path.exists(os.path.dirname(self.total_json_export_path)):
                    os.makedirs(os.path.dirname(self.total_json_export_path))
                    # print(f"Directory {self.total_json_export_path} created.")

                # if the path exists, clear the content in the path
                for file in os.listdir(self.total_image_save_path):
                    os.remove(self.total_image_save_path+'/' + file)
                with open(self.total_json_export_path, 'w') as f:
                    f.write('')

                self.total_images_list = []
                self.total_annotations_list = []
                self.total_categories_list = []

            if self.export_type == 'split':
                # create the path to save the images and annotations
                self.train_image_save_path = os.path.join(split_output_path,'images','Train')
                self.train_json_export_path = os.path.join(split_output_path,'annotations','Train',f'{self.json_name}-Train.json')
                self.test_image_save_path = os.path.join(split_output_path,'images','Test')
                self.test_json_export_path = os.path.join(split_output_path,'annotations','Test',f'{self.json_name}-Test.json')
                # if the path does not exist, create the path
                if not os.path.exists(os.path.dirname(self.train_image_save_path)):
                    os.makedirs(os.path.dirname(self.train_image_save_path))
                if not os.path.exists(os.path.dirname(self.train_json_export_path)):
                    os.makedirs(os.path.dirname(self.train_json_export_path))
                if not os.path.exists(os.path.dirname(self.test_image_save_path)):
                    os.makedirs(os.path.dirname(self.test_image_save_path))
                if not os.path.exists(os.path.dirname(self.test_json_export_path)):
                    os.makedirs(os.path.dirname(self.test_json_export_path))

                # if the path exists, clear the content in the path

                for file in os.listdir(self.train_image_save_path):
                    os.remove(self.train_image_save_path +'/' + file)

                with open(self.train_json_export_path, 'w') as f:
                    f.write('')

                for file in os.listdir(self.test_image_save_path):
                    os.remove(self.test_image_save_path +'/' + file)

                with open(self.train_json_export_path, 'w') as f:
                    f.write('')

                self.train_list = []
                self.test_list = []
    
    def build_image_path_dict(self):
        '''Function to build a dictionary for the image paths
        Args:
            `image_source_paths`: the paths of the images
        Returns:
            `image_dict`: the dictionary of the image paths
        '''
        for source_path in self.image_source_paths:
            for f in os.listdir(source_path):
                # 假设 f 是唯一的，否则要用多值存储结构
                self.image_dict[f] = os.path.join(source_path, f)
        return self.image_dict

    def calculate_bbox(self, center_piont, w, h):
        '''Function to calculate the bounding box of the cropped image
        Args:
            `center_piont`: the center point of the object
            `w`: the width of the cropped image
            `h`: the height of the cropped image
        Returns:
            `x1`, `y1`, `x2`, `y2`: the coordinates of the bounding box(x1, y1, x2, y2), which are the top-left and bottom-right points of the bounding box. This is different from (x, y, w, h)!!!
        '''
        x, y = center_piont
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return x1, y1, x2, y2

    def cut_with_bbox(self, raw_image,x1, y1, x2, y2,file_name):
        '''Function to crop the image with the bounding box
        Args:
            `raw_image`: the original image
            `x1`, `y1`, `x2`, `y2`: the coordinates of the bounding box(x1, y1, x2, y2), which are the top-left and bottom-right points of the bounding box. This is different from (x, y, w, h)!!!
            file_name: the name of the image file
        Returns:
            `cropped_image`: the cropped image
            `cropped_flag`: a flag to indicate whether the cropping is successful. If the cropping is successful, the flag is True. Otherwise, the flag is False.
        '''
        height, width, _ = raw_image.shape
        if x1< 0 or y1 < 0 or x2 > width or y2 > height:
            # print(f"Invalid crop area for origin image: {file_name}")
            # print(f"Image size: {width}x{height}")
            # print(f"Crop area: {x1}, {y1}, {x2}, {y2}")
            expand_image = cv2.copyMakeBorder(raw_image, height, height, width, width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            cropped_image = expand_image[int(y1)+height:int(y2)+height, int(x1)+width:int(x2)+width]
            cropped_flag = True
            # print(f"Successfully crop the {file_name} after expanding the image")
            return cropped_image, cropped_flag
        else:
            cropped_image = raw_image[int(y1):int(y2), int(x1):int(x2)]
            cropped_flag = True
            return cropped_image, cropped_flag
        
    def keypoints_tranformation(self, keypoints, x1, y1):
        ''' Function to transform the keypoints
        Args:
            `keypoints`: the keypoints of the object
            `x1`, `y1`: the top-left point of the bounding box
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

    def expand_bbox(self, image_info, raw_bbox, scale = None, bbox_width = None, bbox_height = None):
        ''' Function to expand the bounding box with a scale or the specified width and height.
        Args:
            `image_info`: the information of the image (from json file)
            `raw_bbox`: list, [x1, y1, w, h]
            `scale`: float, the scale to expand the bbox
            `bbox_width`: float, the width of the bounding box
            `bbox_height`: float, the height of the bounding box
        Returns:
            `new_x1`, `new_y1`, `new_w`, `new_h`: float, the new bounding box
        '''
        x1, y1, w, h = raw_bbox
        x2 = x1 + w
        y2 = y1 + h
        # calculate the center of the bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if scale == None and bbox_width != None and bbox_height != None:

            # get the witdth and height of the image
            image_width = image_info['width']
            image_height = image_info['height']
            new_x1 = center_x - bbox_width / 2
            new_y1 = center_y - bbox_height / 2

            # if the new bounding box is out of the image, adjust the bounding box
            if new_x1 < 0:
                new_x1 = 0
            if new_y1 < 0:
                new_y1 = 0
            if new_x1 + bbox_width > image_width:
                new_x1 = image_width - bbox_width
            if new_y1 + bbox_height > image_height:
                new_y1 = image_height - bbox_height
            return new_x1, new_y1, bbox_width, bbox_height

        elif scale != None and bbox_width == None and bbox_height == None:
            new_w = w * scale
            new_h = h * scale
            new_x1 = center_x - new_w / 2
            new_y1 = center_y - new_h / 2
            return new_x1, new_y1, new_w, new_h
        
        else:
            print('Please specify the scale or the width and height')
            return

    def update_categories(self, values):
        '''Function to update the list of categories, images, and annotations
        Args:
            `values`: the values to be updated
            `list_type`: the type of the list
        '''
        for value in values:
            if value['name'] not in self.name_of_categories:
                value['id'] = len(self.name_of_categories) + 1
                self.name_of_categories.append(value['name'])
                self.dataset_categories_list.append(value)
            else:
                print(f"Category {value['name']} already exists in the dataset")


    # match and return the category id 
    def get_catergory_id(self, category_name):
        '''Function to match and return the category id
        Args:
            category_name: the name of the category
        Returns:    
            the category id
        '''
        for i in range(len(self.dataset_categories_list)):
            if self.dataset_categories_list[i]['name'] == category_name:
                return self.dataset_categories_list[i]['id']

    def setup_categories(self,id,name,supercategory,keypoints,skeleton):
        '''Function to setup the categories of the dataset manually
            A single category is a dictionary with the following:

                "categories": [
                {
                    "id": 1,
                    "name": "myfish",
                    "supercategory": "",
                    "keypoints": [
                        "head",
                        "body",
                        "joint",
                        "tail"
                    ],
                    "skeleton": [
                        [
                            3,
                            2
                        ],
                        [
                            1,
                            2
                        ],
                        [
                            3,
                            4
                        ]
                    ]
                }
            ],

        Args:
            `id`: the id of the category
            `name`: the name of the category
            `supercategory`: the supercategory of the category
            `keypoints`: the keypoints of the category
            `skeleton`: the skeleton of the category
        '''
        self.dataset_categories_list.append({"id": id, "name": name, "supercategory": supercategory, "keypoints": keypoints, "skeleton": skeleton})

    

    def load_from_temp_source(self, temp_json_source_path, temp_image_source_path, load_type):
        '''Function to load the images and annotations from the temp source
        Args:
            `temp_json_source_path`: the path of the json files
            `temp_image_source_path`: the path of the image files
            `load_type`: the type of the loading task
        '''

        # make sure the same path would not be loaded twice
        if temp_json_source_path in self.json_source_paths:
            print(f"Warning: {temp_json_source_path} has already been loaded.")
            return
        
        if temp_image_source_path in self.image_source_paths:
            print(f"Warning: {temp_image_source_path} has already been loaded.")
            return
        
        self.json_source_paths.append(temp_json_source_path)
        self.image_source_paths.append(temp_image_source_path)

        # case1: load the images and annotations from the customize
        if load_type == 'customize':

            # load the images filenames
            all_image_files_dir = [f for f in os.listdir(temp_image_source_path) 
                    if f.lower().endswith(('png', 'jpg', 'jpeg'))]

            # load json files
            all_json_files = [f for f in os.listdir(temp_json_source_path) 
                    if f.lower().endswith(('json'))]
            pbar = tqdm.tqdm(total=100)
            for json_file in all_json_files:

                # Check out whether the filename of the json file could match the image file from `all_image_files_dir`. 
                json_basename = os.path.splitext(json_file)[0]
                matched = any(os.path.splitext(f)[0] == json_basename for f in all_image_files_dir)
                # If not, exclude the json file from `all_json files` and print out the warning message
                if not matched:
                    print(f"Warning: no matching image file for {json_file}, exclusion of JSON file.")
                    continue  
                # If matched, load the json file 
                json_input_path = os.path.join(temp_json_source_path, json_file)
                # and load the image file
                image_input_path = os.path.join(temp_image_source_path, json_basename + '.png')
                # read the image file
                image = cv2.imread(image_input_path)
                # get the width and height of the image
                h, w, _ = image.shape

                # read the json file
                with open(json_input_path, 'r') as f:
                    data = json.load(f)
                keypoints = data[0]['keypoints']
                bbox = data[0]['bbox'][0]
                annotation_unit = {"id": self.counter,
                    "image_id": self.counter,
                    "category_id": 1, # if there are multiple categories, category_id should not be static!!!!!
                    "segmentation": [],
                    "area": h * w, 
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {
                        "occluded": 'false',
                        "track_id": 0,
                        "keyframe": 'true'
                    },
                    "keypoints": [
                        keypoints[0][0],
                        keypoints[0][1],
                        2,
                        keypoints[1][0],
                        keypoints[1][1],
                        2,
                        keypoints[2][0],
                        keypoints[2][1],
                        2,
                        keypoints[3][0],
                        keypoints[3][1],
                        2
                    ],
                    "num_keypoints": 4
                }
                image_unit = {"id": self.counter, 
                              "file_name": json_basename + '.png', 
                              "width": w, 
                              "height": h, 
                              "date_captured":0 , 
                              "license": 1, 
                              "coco_url": "", 
                              "flickr_url": ""}
                # create a list for the json file and its image 
                json_unit = [image_unit, annotation_unit]
                self.total_list.append(json_unit)
                self.counter += 1
                pbar.update(100/len(all_json_files))
            pbar.close()

            print(f'load_from_temp_source {temp_json_source_path} with {load_type} export_type is done')


        # case2: load the images and annotations from the cvat
        if load_type == 'cvat':
            # load the sigle json file
            with open(temp_json_source_path, 'r') as f:
                data = json.load(f)
            # load the categories and update the `dataset_categories_list`
            # initialize the dictionary each time
            category_id_to_name = {}
            # create a dictionary to store the category id and category name
            category_id_to_name = {category['id']: category['name'] for category in data['categories']}
            self.update_categories(data['categories'])
            # iterate through all the annotations
            pbar = tqdm.tqdm(total=100)
            for i in range(len(data['annotations'])):
                annotation_unit = data['annotations'][i]
                # get the unique id
                unique_id = annotation_unit['id']
                # find the corresponding id in `data['images']`
                for image in data['images']:
                    if image['id'] == unique_id:
                        image_unit = image
                        # print(f"image_unit: {image_unit} has been found.")
                        break
                # modify the annotation information
                annotation_unit['id'] = self.counter
                image_unit['id'] = self.counter
                annotation_unit['image_id'] = self.counter
                # get the category name
                category_name = category_id_to_name.get(annotation_unit['category_id'], None)
                # update category id by the category name above
                annotation_unit['category_id'] = self.get_catergory_id(category_name)
                # create a list for the json file and its image 
                json_unit = [image_unit, annotation_unit]
                self.total_list.append(json_unit)
                self.counter += 1
                pbar.update(100/len(data['annotations']))
            pbar.close()
            print(f'load_from_temp_source {temp_json_source_path} with {load_type} export_type is done')


    def finish_loding(self):
        '''Function to finish the loading task.
            This function is used to accomplish the building task of the image dictionary, and it should be called after the loading task is done.
        '''

        self.build_image_path_dict()

        print('Accomplished the building task of the image dictionary.')


    def split_dataset(self,train_ratio):
        '''Function to split the dataset into the training and testing sets.
        This is a temporary version, and it would not consider the varity of the labels.
        Args:
            `train_ratio`: the ratio of the training set
        '''
        # calculate the number of images in the training set
        num_train = math.ceil(len(self.total_list) * train_ratio)
        # shuffle the list
        np.random.shuffle(self.total_list)
        # split the list
        self.train_list = self.total_list[:num_train]
        self.test_list = self.total_list[num_train:]
        print(f"Number of images in the training set: {len(self.train_list)}")
        print(f"Number of images in the testing set: {len(self.test_list)}")

    def crop_pipeline(self,output_width,output_height,image_info,annotation_info,export_image_path = None,draw_keypoint_flag = False,show_flag = False):
        '''Function to crop the an image
        Args:
            `output_width`: the width of the cropped image
            `output_height`: the height of the cropped image
            `image_info`: the information of the image
            `annotation_info`: the information of the annotation
            `export_image_path`: the path to save the image
            `draw_keypoint_flag`: the flag to draw the keypoints on the image
            `show_flag`: the flag to show the image
        '''
        #select keypoint "body" as the center point
        x = annotation_info['keypoints'][3]
        y = annotation_info['keypoints'][4]
        center_point = (x, y)
        x1, y1, x2, y2 = self.calculate_bbox(center_point, output_width, output_height)
        # find the file name of the image
        file_name = image_info['file_name']
        # get the path of the image
        image_path = self.image_dict[file_name]
        # read the original image
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            print(f"Image not found: {image_path}")
            return False
        # crop the image
        cropped_image,cropped_flag = self.cut_with_bbox(raw_image, x1, y1, x2, y2,file_name)
        # modify the annotation information
        new_keypoints = self.keypoints_tranformation(annotation_info['keypoints'], x1, y1)
        annotation_info['bbox'] = 0,0,output_width,output_height
        annotation_info['keypoints'] = new_keypoints
        annotation_info['area'] = output_width*output_height
        image_info['width'] = output_width
        image_info['height'] = output_height

        # update the annotation information
        if cropped_flag == False:
            print(f"Fail to crop the {file_name}, skip this image")
            return False
        else:
            # print(f"Successfully crop the {file_name}")
            if draw_keypoint_flag == True:
                # draw the keypoints on the cropped image
                for i in range(0, len(new_keypoints), 3):
                    x = new_keypoints[i]
                    y = new_keypoints[i+1]
                    v = new_keypoints[i+2]
                    if v == 2:
                        cv2.circle(cropped_image, (int(x), int(y)), 5, (0, 255, 0), -1)   
            if show_flag == True:
                cv2.imshow('sample-test.png', cropped_image)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()
            if export_image_path != None:
                cv2.imwrite(os.path.join(export_image_path, file_name), cropped_image)
            return True


    def expanse_pipeline(self,raw_bbox,image_info,annotation_info,scale = None, width=None,height=None, export_image_path = None, draw_keypoint_flag = False,darw_bbox_flag = False,show_flag = False):
        '''Function to expanse the bounding box of the image
        Args:
            `raw_bbox`: the raw bounding box of the image
            `scale`: the scale to expanse the bounding box
            `image_info`: the information of the image
            `annotation_info`: the information of the annotation
            `export_image_path`: the path to save the image
            `draw_keypoint_flag`: the flag to draw the keypoints on the image
            `darw_bbox_flag`: the flag to draw the bounding box on the image
            `show_flag`: the flag to show the image
        '''
        # expand the bounding box
        new_x1, new_y1, new_w, new_h = self.expand_bbox(image_info, raw_bbox, scale = scale,bbox_width=width,bbox_height=height)
        # modify the annotation information
        annotation_info['bbox'] = new_x1, new_y1, new_w, new_h
        # export the image
        
        # get the file name of the image
        file_name = image_info['file_name']
        # get the path of the image
        image_path = self.image_dict[file_name]
        # read the original image
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            print(f"Image not found: {image_path}")
            return False
            
        if draw_keypoint_flag == True or darw_bbox_flag == True:
            if draw_keypoint_flag == True:
                # draw the keypoints on the raw image
                for i in range(0, len(annotation_info['keypoints']), 3):
                    x = annotation_info['keypoints'][i]
                    y = annotation_info['keypoints'][i+1]
                    v = annotation_info['keypoints'][i+2]
                    if v == 2:
                        cv2.circle(raw_image, (int(x), int(y)), 5, (0, 255, 0), -1)
            if darw_bbox_flag == True:
                # draw the new bounding box on the raw image
                x1, y1, w, h = annotation_info['bbox']
                x2 = x1 + w
                y2 = y1 + h
                cv2.rectangle(raw_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if show_flag == True:
            cv2.imshow('sample-test.png', raw_image)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()

        if export_image_path != None:
        # save the raw image
            cv2.imwrite(os.path.join(export_image_path, file_name), raw_image)

            

    def export_dataset_pipeline(self,list_of_units,export_image_path,export_json_path,mode,output_width = None, output_height = None, scale = None,bbox_width = None, bbox_hight = None, draw_keypoint_flag = False,darw_bbox_flag = False,show_flag = False):
        '''Function to export a single dataset (or a subset) to the target directory
            Note: This function might be called multiple times to export multiple datasets (or subsets) to the target directory.
        Args:
            `list_of_units`: the list of units, each unit contains the information of the image and its annotation
            `export_image_path`: the path to save the images
            `export_json_path`: the path to save the json file
            `mode`: the mode of the exportation
            `output_width`: the width of the cropped image
            `output_height`: the height of the cropped image
            `scale`: the scale to expanse the bounding box
            `bbox_width`: the width of the bounding box
            `bbox_hight`: the height of the bounding box
            `draw_keypoint_flag`: the flag to draw the keypoints on the image
            `darw_bbox_flag`: the flag to draw the bounding box on the image
            `show_flag`: the flag to show the image
        '''
        # initialize the lists of images and annotations
        images_list = []
        annotations_list = []
        pbar = tqdm.tqdm(total=100)
        for unit in list_of_units:
            # get the image and annotation information
            image_info = unit[0]
            annotation_info = unit[1]
            images_list.append(image_info)
            annotations_list.append(annotation_info)

            # export the image
            if mode == 'original':
                # setup the export path. Find the image path from `image_source_paths` by using its unique file name

                # # method 1: iterate through the `image_source_paths`
                # for image_source_path in self.image_source_paths:
                #     if item['file_name'] in os.listdir(image_source_path):
                #         import_image_path = os.path.join(image_source_path, item['file_name'])
                #         print(f'import_image_path: {import_image_path}')
                #         break

                # method 2: use the dictionary that has been built
                import_image_path = self.image_dict[image_info['file_name']]
                # copy the image to the target directory
                shutil.copyfile(import_image_path, os.path.join(export_image_path,image_info['file_name']))

            if mode == 'cropped':
                if output_width == None or output_height == None:
                    print('Please specify the output width and height')
                    return
                # crop the image
                self.crop_pipeline(output_width,output_height,image_info,annotation_info,export_image_path,draw_keypoint_flag =  draw_keypoint_flag ,show_flag = show_flag)

            if mode == 'expanse':
                # expanse the bounding box
                self.expanse_pipeline(annotation_info['bbox'],image_info,annotation_info,scale = scale,width= bbox_width, height=bbox_hight,export_image_path =export_image_path, draw_keypoint_flag= draw_keypoint_flag ,darw_bbox_flag= darw_bbox_flag,show_flag= show_flag)

            pbar.update(100/len(list_of_units))
        pbar.close()
        # export the json file with annotations
        data = {
        "categories": self.dataset_categories_list, # the categories of the dataset are always the same
        "images": images_list,
        "annotations": annotations_list
        }
        with open(export_json_path, 'w') as f:
            json.dump(data, f, indent=4)
        # cope the images to the target directory
        # for item in images_list:
            
        print(f'export_dataset with {mode} is done')
        print(f'Number of images: {len(images_list)}', end='\n')
    
    def export_original(self):
        '''Function to export the original dataset
        '''
        self.export_dataset_pipeline(self.total_list,self.total_image_save_path,self.total_json_export_path,'original')
    
    def export_cropped(self,output_width,output_height):
        '''Function to export the cropped dataset
        Args:
            `output_width`: the width of the cropped image
            `output_height`: the height of the cropped image
        '''
        if self.train_list == [] or self.test_list == []:
            print('Please split the dataset first')
            return
        else:
            self.export_dataset_pipeline(self.train_list,self.train_image_save_path,self.train_json_export_path,'cropped',output_width,output_height,draw_keypoint_flag=False)
            self.export_dataset_pipeline(self.test_list,self.test_image_save_path,self.test_json_export_path,'cropped',output_width,output_height,draw_keypoint_flag=False)

    def try_cropped(self,w,h,num):
        '''Function to try the cropped function
        Args:
            `w`: the width of the cropped image
            `h`: the height of the cropped image
            `num`: the index of the image in the list `total_list`
        '''
        self.crop_pipeline(w,h,self.total_list[num][0],self.total_list[num][1],export_image_path = None,draw_keypoint_flag = True,show_flag = True)

    def export_expanse(self,scale = None, bbox_width = None, bbox_hight = None):
        '''Function to expanse the bounding box of the image
        Args:
            `scale`: the scale to expanse the bounding box
            `bbox_width`: the width of the bounding box
            `bbox_hight`: the height of the bounding box
        
        Do not provide both a `scale` and bounding box dimensions (`bbox_width`, `bbox_height`) at the same time.You must specify either a scale value or the exact width and height for the bounding box.
        '''
        if self.train_list == [] or self.test_list == []:
            print('Please split the dataset first')
            return
        else:
            self.export_dataset_pipeline(self.train_list,self.train_image_save_path,self.train_json_export_path,'expanse',scale = scale,bbox_width = bbox_width, bbox_hight = bbox_hight)
            self.export_dataset_pipeline(self.test_list,self.test_image_save_path,self.test_json_export_path,'expanse',scale = scale,bbox_width = bbox_width, bbox_hight = bbox_hight)
    
    def try_expanse(self,scale,num):
        '''Function to try the expanse function
        Args:
            `scale`: the scale to expanse the bounding box
            `num`: the index of the image in the list `total_list`
        '''
        self.expanse_pipeline(self.total_list[num][1]['bbox'],scale,self.total_list[num][0],self.total_list[num][1],draw_keypoint_flag=True,darw_bbox_flag=True,show_flag=True)

#################################################################
'''
The following functions are the examples of how to use the class `PeterDataset` and its member functions to create a pipeline to handle the dataset.
'''
def pipeline_export_total():
    # create the dataset
    my_total_output_path = '/home/peter/Desktop/Fish-Dataset/Fish-0223/2CVAT/fish-0223-demo1'
    my_json_name = 'fish-0223-demo1'
    dataset = PeterDataset(save_flag = True, export_type= 'total',json_name=my_json_name,total_output_path=my_total_output_path,split_output_path = None)
    # setup the categories
    dataset.setup_categories(1,"myfish"," ",["head","body","joint","tail"],[[3,2],[1,2],[3,4]])

    # load the images and annotations from the temp source
    json_source_path_1 = '/home/peter/Desktop/Fish-Dataset/Fish-0223/model_outcome/fish-0223-demo1/annotations'
    image_source_path_1 = '/home/peter/Desktop/Fish-Dataset/Fish-0223/extract_outcome/fish-0223-demo1'
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

    # finish the loading task
    dataset.finish_loding()

    # check the cropping outcome directly
    dataset.try_cropped(256,256,667)
###############################################################

def formatting_json(json_input_path):
    '''Function to format the json file
    Args:
        `json_input_path`: the path of the json file
    
    The json file that export from CVAT is a single line json file, which is not readable. This function is used to format the json file.
    '''
    with open(json_input_path, 'r') as f:
        data = json.load(f)
    # print(data)
    with open(json_input_path, 'w') as f:
        json.dump(data, f, indent=4)


def main():

    # json_input_path = '/home/peter/Desktop/Fish-Dataset/Fish-0223/model_outcome/fish-0223-demo1/annotations/fish-0223-demo1_frame_000003.json'
    # source_dir = '/home/peter/Desktop/Fish-Dataset/Fish-0223/model_outcome/fish-0223-demo1/annotations'
    # read_single_json(json_input_path)
    # read_multiple_json(source_dir)


    # formatting_json('/home/peter/Desktop/Fish-Dataset/Fish-0223/real_annotations/fish-0223-demo1.json')

    pipeline_export_total()
    # pipeline_export_corpped()
    # pipeline_check_cropped()

if __name__ == '__main__':
    main()

'''
The following section contains some old functions that are not used in the current version of the class `PeterDataset`.
'''

# def read_multiple_json(source_dir):
#     all_json_files = [f for f in os.listdir(source_dir) 
#                    if f.lower().endswith(('json'))]
#     print(len(all_json_files))

# def read_single_json(json_input_path):
#     with open(json_input_path, 'r') as f:
#         data = json.load(f)
#     keypoints = data[0]['keypoints']
#     print(keypoints)
#     # print(keypoints[0][0])
#     # print(keypoints[0][1])
#     # print(keypoints[1][0])
#     bbox = data[0]['bbox'][0]
#     print(bbox)
#     test_unit = {"image_id": 0, "category_id": 1, "keypoints": [keypoints[0][0],keypoints[0][1],2,keypoints[1][0],keypoints[1][1],2,keypoints[2][0],keypoints[2][1],2,keypoints[3][0],keypoints[3][1],2], "num_keypoints": 4, "bbox": bbox, "area": 0, "iscrowd": 0}
#     print(test_unit)