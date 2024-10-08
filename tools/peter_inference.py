from mmdet.apis import DetInferencer

# 初始化模型
inferencer = DetInferencer(model='/home/peter/mmdetection/configs/fish/peter-rtmdet_tiny_8xb32-300e_coco.py', weights='/home/peter/mmdetection/work_dirs/peter-rtmdet_tiny_8xb32-300e_coco/epoch_300.pth',
device='cuda:0')
img_path = '/home/peter/Desktop/Fish-Dataset/Fish-1001/goldfish2.png'
inferencer(img_path, out_dir='/home/peter/Desktop/Fish-Dataset/test-output-0929')