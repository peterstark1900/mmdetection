import os
import cv2

def extract_frames_from_videos(source_dir, dest_dir):
    # 如果目标目录不存在则创建
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # 遍历目录A下的所有视频文件（支持 .mp4, .avi, .mov, .mkv ）
    video_files = [f for f in os.listdir(source_dir) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    for video_file in video_files:
        video_path = os.path.join(source_dir, video_file)
        # 使用去掉扩展名的视频名称，作为目标子文件夹名称
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(dest_dir, video_name)
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 保存当前帧，文件名格式为 frame_000000.png
            frame_filename = os.path.join(video_output_dir, f"{video_name}_frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_idx += 1
        cap.release()
        print(f"Finished processing video: {video_file}, extracted {frame_idx} frames.")

def main():
    # 示例调用，将目录A下的视频帧导出到目录B（请根据实际路径修改）
    source_directory = '/home/peter/Desktop/Fish-Dataset/Fish-0223/original'
    destination_directory = '/home/peter/Desktop/Fish-Dataset/Fish-0223/temp-version'
    extract_frames_from_videos(source_directory, destination_directory)

if __name__ == '__main__':
    main()