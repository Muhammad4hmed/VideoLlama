import os
import argparse
import json
import torch
from tqdm import tqdm
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.eval.inference import video_chatgpt_infer
import pickle
import warnings
warnings.filterwarnings("ignore")
from metrics import get_scores

def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--feature_dir', help='Directory containing video files.', required=False, default = "/home/waleed/ahmed/LLM/Video-ChatGPT/features_breakfast_fawad_final-v2")
    parser.add_argument('--gt_dir', help='Directory containing ground truths.', required=False, default = "/home/waleed/ahmed/LLM/Breakfast/Videos/BreakfastII_15fps_qvga_sync")
    #parser.add_argument('--gt_dir', help='Directory containing ground truths.', required=False, default = "/home/waleed/ahmed/LLM/Breakfast/Breakfast")
    
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=False, default='Predictions')
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=False, default="Predictions_Breakfast")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--print", type=bool, required=False, default=False)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--projection_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=False, default="valid.txt")

    return parser.parse_args()

def load_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model, tokenizer, video_token_len = initialize_model(args.model_name, args.projection_path)
    # Load the ground truth file
    # with open(args.gt_file) as file:
    #     gt_contents = json.load(file)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    vid_features = []
    gts = []
    seg_sizes = []
    vid_names = []
    actions = []
    total_frame_num = []
    act_total_frame_num = []
    frames = load_dict_from_file('vid_frames_breakfast.json')
    with open(args.valid_path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n','')
            name = line.split('/')[-1].split('.')[0] + '_' + line.split('/')[-2]
            vid_names.append(name)
            actions.append(" ".join(line.split('/')[-1].split('.')[0].split('_')[1:]))
            total_frames = frames[name]
            act_total_frame_num.append(total_frames)
            seg_size = float(total_frames - 1) / 1000
            seg_sizes.append(seg_size)
            total_frame_num.append(int(total_frames/seg_size))
            video = args.feature_dir + '/' + name +'.pkl'
            #print(video)
            with open(video, "rb") as f1:
                features = pickle.load(f1)
                vid_features.append(features)
            f1.close()

            gt = args.gt_dir + '/' + line + '.labels'
            #print(gt)
            if not os.path.exists(gt): gt = gt.replace('ch0', 'ch1')
            with open(gt, "r") as f1:
                truth = []
                gts.append("\n".join(f1.readlines()))
            f1.close()
            
    f.close()
    i = 0
    pp=0
    for sample in tqdm(vid_features):
        sample_out = {}
        #question = f'List down all the segments. action is {actions[i]}.'# maximum frames are {act_total_frame_num[i]}'
        question= 'What are the segments in the video and their frames ?'
        # Run inference on the video and add the output to the list
        torch.cuda.set_device(1)

        output = video_chatgpt_infer(torch.from_numpy(sample).cuda(), question, conv_mode, model,
                                             tokenizer, video_token_len, seg_sizes[i], total_frame_num[i])
        # print(len(output))
        sample_out['name'] = vid_names[i]
        sample_out['truth'] = gts[i]
        sample_out['pred'] = output
        
        if args.print:
            print(question)
            print('Ground truth', gts[i])
            print('Prediction', output)

        output_list.append(sample_out)

        if (i > 0) and (i % 10 == 0):
            save_path = os.path.join(args.output_dir, f"{args.output_name}.json")
            with open(save_path, 'w') as file:
                json.dump(output_list, file)

            get_scores(save_path)

        i += 1
        
    # Save the output list to a JSON file
    save_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    with open(save_path, 'w') as file:
        json.dump(output_list, file)

    get_scores(save_path)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)