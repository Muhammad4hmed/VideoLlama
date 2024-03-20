import cv2
import os

def break_video_into_chunks(video_filename, video_path, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    chunk_count = 1
    current_frame = 0
    while current_frame < frame_count:

        start_frame = current_frame
        end_frame = min(current_frame + 5000, frame_count)

        output_chunk_path = os.path.join(output_folder, f"{video_filename}_{chunk_count}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_chunk_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                out.release()
                cap.release()
                return
            out.write(frame)
            current_frame += 1

        out.release()
        chunk_count += 1

    cap.release()

train_videos_folder = "/home/waleed/ahmed/LLM/Epic Kitchens Stripped/3h91syskeag572hl6tvuovwv4d/videos/train"
person_folders = os.listdir(train_videos_folder)
person_folders.sort()
skip_list = ['P01_01.MP4']

for person_folder in person_folders:
    person_folder_path = os.path.join(train_videos_folder,person_folder)
    video_files = os.listdir(person_folder_path)
    video_files.sort()
    
    for video_file in video_files:
        if video_file in skip_list:
            continue

        video_path = os.path.join(person_folder_path, video_file)
    
        output_folder = os.path.join("/home/waleed/ahmed/LLM/Epic Kitchens Divided/3h91syskeag572hl6tvuovwv4d/videos/train", person_folder)
        break_video_into_chunks(video_file.removesuffix('.MP4'), video_path, output_folder)

print("Video chunking complete.")
