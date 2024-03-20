import os

train_videos_folder = "/home/waleed/ahmed/LLM/Epic Kitchens Divided/3h91syskeag572hl6tvuovwv4d/videos/train"
# person_folders = os.listdir(train_videos_folder)
# person_folders.sort()
person_folders = ['P01']
skip_list = ['P01_01_1', 'P01_01_2', 'P01_01_3', 'P01_01_4', 'P01_01_5', 'P01_01_6', 'P01_01_7', 'P01_01_8', 'P01_01_9', 'P01_01_10', 'P01_01_11', 'P01_01_12', 'P01_01_13', 'P01_01_14', 'P01_01_15', 'P01_01_16', 'P01_01_17', 'P01_01_18', 'P01_01_19', 'P01_01_20', 'P01_02_1', 'P01_02_2', 'P01_02_3', 'P01_02_4', 'P01_02_5', 'P01_02_6', 'P01_02_7', 'P01_03_1', 'P01_03_2', 'P01_04_1', 'P01_04_2', 'P01_05_1', 'P01_05_2', 'P01_05_3', 'P01_05_4', 'P01_05_5', 'P01_05_6', 'P01_05_7', 'P01_05_8', 'P01_05_9', 'P01_05_10', 'P01_05_11', 'P01_05_12', 'P01_05_13', 'P01_05_14', 'P01_05_15', 'P01_05_16', 'P01_06_1', 'P01_06_2', 'P01_06_3', 'P01_06_4', 'P01_06_5', 'P01_06_6', 'P01_07_1', 'P01_07_2', 'P01_08_1', 'P01_08_2', 'P01_09_1', 'P01_09_2', 'P01_09_3', 'P01_09_4', 'P01_09_5', 'P01_09_6', 'P01_09_7', 'P01_09_8', 'P01_09_10', 'P01_09_11', 'P01_09_12', 'P01_09_13', 'P01_09_14']

for person_folder in person_folders:
    person_folder_path = os.path.join(train_videos_folder,person_folder)
    video_files = os.listdir(person_folder_path)
    video_files.sort()
    
    for video_file in video_files:
        if video_file.removesuffix('.mp4') in skip_list:
            continue

        video_path = os.path.join(person_folder_path, video_file)
    
        output_folder = os.path.join("/home/waleed/ahmed/LLM/Epic Kitchens Divided Compressed", person_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        video_name = video_file.removesuffix('.mp4')
        ffmpeg_command = f'ffmpeg -i "{video_path}" -vcodec libx265 -crf 28 "{output_folder}/{video_name}.mp4"'
    
        os.system(ffmpeg_command)