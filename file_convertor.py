import os
from pathlib import Path
from moviepy.editor import *
# video = VideoFileClip("dataset/Le2i_Fall_Detection_Dataset/home/Videos/video (24).avi")    # 讀取影片

format_list = ['mp4']  # 要轉換的格式清單
folder_dir = Path('dataset/Le2i_Fall_Detection_Dataset/home/Videos')
for vid in folder_dir.glob("*"):
    # print(vid)
    # print(str(vid).replace('.avi', ''))
# 使用 for 迴圈轉換成所有格式
    video = VideoFileClip(str(vid)) 
    for i in format_list:
        output = video.copy()
        output.write_videofile(f"{str(vid).replace('.avi', '')}.{i}",temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")

# print('ok')