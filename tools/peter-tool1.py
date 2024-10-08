import math
import json
import cv2
import json

def calculate_rectangle(x1, y1, x2, y2, w, h):
    # 计算线段的方向向量
    dx = x2 - x1
    dy = y2 - y1
    
    # 计算线段的长度
    L = math.sqrt(dx**2 + dy**2)
    
    # 计算单位方向向量
    ux = dx / L
    uy = dy / L
    
    # 计算垂直于线段的单位方向向量
    vx = -uy
    vy = ux
    
    # 计算矩形的四个顶点
    p1 = (x1 - w/2 * vx, y1 - w/2 * vy)
    p2 = (x1 + w/2 * vx, y1 + w/2 * vy)
    p3 = (x2 - w/2 * vx, y2 - w/2 * vy)
    p4 = (x2 + w/2 * vx, y2 + w/2 * vy)
    
    return p1, p2, p3, p4

def find_max_bbox_size(input_file):
    # 读取 JSON 文件
    with open(input_file, 'r') as f:
        data = json.load(f)

    max_width = 0
    max_height = 0

    for annotation in data['annotations']:
        bbox = annotation['bbox']
        width = bbox[2]
        height = bbox[3]

        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height

    return int(max_width), int(max_height)
def cut_with_max_bbox(input_file, output_file,input_images_file_path,output_images_file_path):
    # 读取 JSON 文件
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 找到最大 bbox 尺寸
    # max_width, max_height = find_max_bbox_size(input_file)
    max_width, max_height = 80,80

    for annotation in data['annotations']:
        bbox = annotation['bbox']
        x, y, width, height = bbox

        # 调整左上角坐标以保持中心点不变
        new_x = int(x - (max_width - width) / 2)
        new_y = int(y - (max_height - height) / 2
)
        # 生成新的 bbox
        new_bbox = [new_x, new_y, max_width, max_height]
        # print(bbox)
        # print(new_bbox)
        
        annotation['bbox'] = [0, 0, max_width, max_height]

        # 计算新的面积
        new_area = max_width * max_height
        annotation['area'] = new_area

        # 计算原 bbox 的中心点
        cx = x + width / 2
        cy = y + height / 2

        # 计算新的 bbox 的中心点
        new_cx = new_x + max_width / 2
        new_cy = new_y + max_height / 2
        image_id_to_file_name = {image['id']: image['file_name'] for image in data['images']}
        image_id = annotation['image_id']
        images = data['images']
        file_name = image_id_to_file_name.get(image_id, None)
        # if file_name:
            # print(f"Annotation ID: {annotation['id']}, Image ID: {image_id}, File Name: {file_name}")
        # else:
        #     print(f"Annotation ID: {annotation['id']}, Image ID: {image_id}, File Name: Not Found")
        if not file_name:
            print(f"Annotation ID: {annotation['id']}, Image ID: {image_id}, File Name: Not Found")
            continue

        image_file_path = input_images_file_path+f'{file_name}'
        # 读取图片
        old_image = cv2.imread(image_file_path)
        if old_image is None:
            print(f"Image not found: {image_file_path}")
            continue
        # 检查裁剪区域是否在图像边界内
        height, width, _ = old_image.shape
        # print(image_file_path)
        # print(old_image.shape)
        # print(' ')
        if new_x < 0 or new_y < 0 or new_x + max_width > width or new_y + max_height > height:
            print(f"Invalid crop area for image: {file_name}")
            continue
        cropped_image = old_image[new_y:new_y + max_height, new_x:new_x + max_width]
        # 保存裁剪后的图片
        output_image_file_path = output_images_file_path + f'{file_name}'
        cv2.imwrite(output_image_file_path, cropped_image)

        # 转换关键点坐标
        if 'keypoints' in annotation:
            new_keypoints = []
            keypoints = annotation['keypoints']
            for j in range(0, len(keypoints), 3):
                kx = keypoints[j]
                ky = keypoints[j + 1]
                visibility = keypoints[j + 2]

                # 计算新的关键点坐标
                new_kx = kx - new_x
                new_ky = ky - new_y

                new_keypoints.extend([new_kx, new_ky, visibility])

            annotation['keypoints'] = new_keypoints
    for image in data['images']:
        image['width'] = max_width
        image['height'] = max_height
    # 将修改后的数据写回到 JSON 文件中
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
        
train_input_file = '/home/peter/mmpose/data/Fish-Tracker-1001/annotations/Fish-Tracker-1001-Train.json'
train_inputs_file_path = '/home/peter/mmpose/data/Fish-Tracker-1001/images/Train/'

train_output_file = '/home/peter/mmpose/data/Fish-Tracker-1002/annotations/Fish-Tracker-1002-Train-new_bbox.json'
train_output_images_file_path = '/home/peter/mmpose/data/Fish-Tracker-1002/images/Train/'

test_input_file = '/home/peter/mmpose/data/Fish-Tracker-1001/annotations/Fish-Tracker-1001-Test.json'
test_inputs_file_path = '/home/peter/mmpose/data/Fish-Tracker-1001/images/Test/'

test_output_file = '/home/peter/mmpose/data/Fish-Tracker-1002/annotations/Fish-Tracker-1002-Test-new_bbox.json'
test_output_images_file_path = '/home/peter/mmpose/data/Fish-Tracker-1002/images/Test/'
cut_with_max_bbox(train_input_file, train_output_file, train_inputs_file_path, train_output_images_file_path)
cut_with_max_bbox(test_input_file, test_output_file, test_inputs_file_path, test_output_images_file_path)

# def scale_bbox(input_file, output_file, scale_factor):
#     # 读取 JSON 文件
#     with open(input_file, 'r') as f:
#         data = json.load(f)

#     for i in range(len(data['annotations'])):
#         annotation = data['annotations'][i]
#         bbox = annotation['bbox']
#         x, y, width, height = bbox

#         # 计算新的宽度和高度
#         new_width = width * scale_factor
#         new_height = height * scale_factor

#         # 调整左上角坐标以保持中心点不变
#         new_x = x - (new_width - width) / 2
#         new_y = y - (new_height - height) / 2

#         # 生成新的 bbox
#         new_bbox = [new_x, new_y, new_width, new_height]

#         annotation['bbox'] = new_bbox

#     # 将修改后的数据写回到 JSON 文件中
#     with open(output_file, 'w') as f:
#         json.dump(data, f, indent=4)
# train_input_file = '/home/peter/mmpose/data/Fish-Tracker-1001/annotations/Fish-Tracker-1001-Train.json'
# train_output_file = '/home/peter/mmpose/data/Fish-Tracker-1001/annotations/Fish-Tracker-1001-Train-new_bbox.json'
# test_input_file = '/home/peter/mmpose/data/Fish-Tracker-1001/annotations/Fish-Tracker-1001-Test.json'
# test_output_file = '/home/peter/mmpose/data/Fish-Tracker-1001/annotations/Fish-Tracker-1001-Test-new_bbox.json'
# scale_bbox(train_input_file, train_output_file, 3)
# scale_bbox(test_input_file, test_output_file, 3)

# 假设你的 JSON 文件名为 'Fish-Tracker-1001-Test-detection.json'
# input_file = '/home/peter/mmpose/data/Fish-Tracker-1001/annotations/Fish-Tracker-1001-Test.json'
# output_file = '/home/peter/mmpose/data/Fish-Tracker-1001/annotations/Fish-Tracker-1001-Test-new_bbox.json'
# # 读取 JSON 文件
# with open(input_file, 'r') as f:
#     data = json.load(f)


# annotation = data['annotations'][600]
# image_id = annotation['image_id']
# images = data['images'][600]
# file_name = images['file_name']
# image_file_path = f'/home/peter/mmpose/data/Fish-Tracker-1001/images/Test/{file_name}'
# bbox = annotation['bbox']
# x, y, width, height = bbox

# # 定义比例因子
# scale_factor = 3

# # 计算新的宽度和高度
# new_width = width * scale_factor
# new_height = height * scale_factor

# # 调整左上角坐标以保持中心点不变
# new_x = x - (new_width - width) / 2
# new_y = y - (new_height - height) / 2

# # 生成新的 bbox
# new_bbox = [new_x, new_y, new_width, new_height]
# # 读取原始图片
# image = cv2.imread(image_file_path)

# # 计算截取区域的坐标
# x1, y1 = int(new_x), int(new_y)
# x2, y2 = int(new_x + new_width), int(new_y + new_height)

# # 截取图片
# cropped_image = image[y1:y2, x1:x2]

# # # 保存截取的图片
# cv2.imwrite('/home/peter/Desktop/Fish-Dataset/test-output-0929/result1007.png', cropped_image)

# # # 显示截取的图片
# # cv2.imshow('Cropped Image', cropped_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # keypoints = annotation['keypoints']
# # print(keypoints)
# # print(keypoints[0])


# for i in range(len(data['annotations'])):

#     annotation = data['annotations'][i]
#     bbox = annotation['bbox']
#     x, y, width, height = bbox
#     # 定义比例因子
#     scale_factor = 3

#     # 计算新的宽度和高度
#     new_width = width * scale_factor
#     new_height = height * scale_factor

#     # 调整左上角坐标以保持中心点不变
#     new_x = x - (new_width - width) / 2
#     new_y = y - (new_height - height) / 2

#     # 生成新的 bbox
#     new_bbox = [new_x, new_y, new_width, new_height]

#     # annotation = data['annotations'][i]

#     # keypoints = annotation['keypoints']
#     # new_bbox = calculate_rectangle(keypoints[0], keypoints[1], keypoints[3], keypoints[4], w=100, h=200)

#     # annotation['bbox'] = new_bbox



#     # print(keypoints)
#     # print(keypoints[0])

#     # print(len(data['annotations']))
#     # x, y, width, height = bbox

#     # # 根据image_id读取相应的图像文件
#     # image_id = annotation['image_id']
#     # # 根据image_id查找对应的file_name
#     # file_name = None
#     # for image in data['images']:
#     #     if image['id'] == image_id:
#     #         file_name = image['file_name']
#     #         break
#     # # print(image_id)
#     # image_path = f'fish-bbox-0924-v1/images/Train/{file_name}'
#     # temp_image = cv2.imread(image_path)

#     # # 检查图像是否成功读取
#     # if temp_image is None:
#     #     raise FileNotFoundError(f"Image {image_path} not found")

#     # # 裁剪图像
#     # cropped_image = temp_image[int(y):int(y+height), int(x):int(x+width)]
#     # # 保存或显示裁剪后的图像
#     # cv2.imwrite(f'fish-dataset-0924-v1/fish_{i}_{file_name}', cropped_image)
#     # # # 或者使用以下代码显示裁剪后的图像
#     # # cv2.imshow('Cropped Image', cropped_image)
#     # # cv2.waitKey(0)
#     # # cv2.destroyAllWindows()


# # # 示例
# # x1, y1 = 1, 1
# # x2, y2 = 4, 4
# # w = 2
# # h = 5

# # rectangle_points = calculate_rectangle(x1, y1, x2, y2, w, h)
# # print("矩形的四个顶点坐标为：", rectangle_points)


# # 将修改后的数据写回到 JSON 文件中
# with open(output_file, 'w') as f:
#     json.dump(data, f, indent=4)