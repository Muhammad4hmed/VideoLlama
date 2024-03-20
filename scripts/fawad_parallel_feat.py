import os
import math
import torch
import glob
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import CLIPImageProcessor
from custom_vision_model import CLIPVisionModel
from transformers import AutoImageProcessor, SwinModel
from datasets import load_dataset
import json
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import traceback

load_video_lock = Lock()

def load_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def save_dict_to_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def load_video(vis_path, name, num_frm=1000):
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)
    if os.path.exists('vid_frames_breakfast.json'):
        vid_frames = load_dict_from_file('vid_frames_breakfast.json')
    else:
        vid_frames = {}
    # name = vis_path.split('//')[-1].split('.')[0]
    vid_frames[name] = total_frame_num
    #save_dict_to_file(vid_frames, 'vid_frames_breakfast.json')
    total_num_frm = num_frm  # min(total_frame_num, num_frm)
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    a, H, W, _ = img_array.shape
    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq


def get_spatio_temporal_features(features, num_temporal_tokens=100):
    t, s, c = features.shape

    temporal_tokens = np.mean(features, axis=1)
    padding_size = num_temporal_tokens - t
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')

    spatial_tokens = np.mean(features, axis=0)
    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)

    return sp_features


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--video_dir_path", required=True, help="Path to read the videos from.")
    parser.add_argument("--clip_feat_path", required=True, help="The output dir to save the features in.")
    parser.add_argument("--infer_batch", required=False, type=int, default=32,
                        help="Number of frames/images to perform batch inference.")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    video_dir_path = args.video_dir_path
    clip_feat_path = args.clip_feat_path
    infer_batch = args.infer_batch
    os.makedirs(clip_feat_path, exist_ok=True)

    # Initialize the CLIP model
    # image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16)
    # vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16,
    # low_cpu_mem_usage=True).cuda()
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-large-patch4-window7-224")
    vision_tower = SwinModel.from_pretrained("microsoft/swin-large-patch4-window7-224")
    vision_tower.eval()
    vision_tower = vision_tower.cuda()
    vision_tower=nn.DataParallel(vision_tower)
    # vision_tower.eval()

    all_videos = sorted(glob.glob(video_dir_path + '/*'))
    video_clip_features = {}
    all_features = {}
    counter = 0
    augs = ['flip', 'bright', 'bright_flip', 'contrast', 'contrast_flip', 'crop', 'crop_flip', 'compress',
            'compress_flip', 'sharp', 'sharp_flip']
    global_video_features={}
    for video_name in tqdm(all_videos):
        for cam in tqdm(glob.glob(video_name + '/*')):
    # Inner loops parallelized
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for action in glob.glob(cam + '/*.avi'):
                    futures.append(executor.submit(process_action, action,cam,clip_feat_path,image_processor,infer_batch,vision_tower,video_clip_features))
            for future in as_completed(futures):
             pass
    

  

def process_action(action,cam,clip_feat_path,image_processor,infer_batch,vision_tower,video_clip_features):
    try:
        video_path = action
        video_id = action.split('/')[-1].split('.')[0] + '_' + cam.split('/')[-1]
        if os.path.exists(f"{clip_feat_path}/{video_id}.pkl"):
            load_video(video_path, video_id)
            return None
        # with load_video_lock:
        #     video = load_video(video_path, video_id)
        video = load_video(video_path, video_id)
        video_tensor = image_processor(video, return_tensors="pt")

        n_chunk = len(video_tensor["pixel_values"])
        video_features = torch.FloatTensor(n_chunk, 1024).fill_(0)
        n_iter = int(math.ceil(n_chunk / float(infer_batch)))

        for i in range(n_iter):
            min_ind = i * infer_batch
            max_ind = (i + 1) * infer_batch
            video_batch = video_tensor["pixel_values"][min_ind:max_ind].cuda()

            with torch.no_grad():
                image_forward_outs = vision_tower(video_batch, output_hidden_states=True)

            select_hidden_state = image_forward_outs.pooler_output
            batch_features = select_hidden_state[:, :1024]
            video_features[min_ind:max_ind] = batch_features.detach().cpu()

        # Return the video features and ID
        #return video_id, video_features.numpy().astype("float16")
          
            features = video_features
            # print(features.shape)
            with open(f"{clip_feat_path}/{video_id}.pkl", 'wb') as f:
                pickle.dump(features, f)

    except Exception as e:
        traceback.print_exc()
        print(f"Can't process {video_path}")
        return None


if __name__ == "__main__":
    main()