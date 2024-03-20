import os

train_videos_folder = "/home/waleed/ahmed/LLM/Epic Kitchens/3h91syskeag572hl6tvuovwv4d/videos/train"
person_folders = os.listdir(train_videos_folder)
person_folders.sort()
skip_list = []

for person_folder in person_folders:
    person_folder_path = os.path.join(train_videos_folder,person_folder)
    video_files = os.listdir(person_folder_path)
    video_files.sort()
    
    for video_file in video_files:
        if video_file in skip_list:
            continue

        video_path = os.path.join(person_folder_path, video_file)
    
        output_folder = os.path.join("/home/waleed/ahmed/LLM/Epic Kitchens Stripped/3h91syskeag572hl6tvuovwv4d/videos/train", person_folder)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_path = os.path.join(output_folder, video_file)    
        ffmpeg_command = f'ffmpeg -i \"{video_path}\" -c:v copy -c:a aac -strict experimental \"{output_path}\"'
        os.system(ffmpeg_command)
        