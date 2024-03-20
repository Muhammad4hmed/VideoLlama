import json
import argparse
import glob
import json
import numpy as np
import re
import os
import random
import tqdm

def load_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--input_json_file", required=True, help="")
    parser.add_argument("--output_json_file", required=True, help="")

    args = parser.parse_args()

    return args

def extract_numbers_from_string(string):
    pattern = r'(\d+(\.\d+)?)'
    numbers = re.findall(pattern, string)
    extracted_numbers = [float(match[0]) for match in numbers]
    return extracted_numbers


def divide_numbers_if_starts_with(string, max_frame, picked_frames = 1000):
    divider = np.round((max_frame-1)/picked_frames)
    divider = 1 if divider == 0 else divider
    numbers = extract_numbers_from_string(string)

    if len(numbers) >= 2:
        number1, number2 = numbers[:2]
    elif len(numbers) == 1:
        number1, number2 = numbers[0], numbers[0]
    else:
        return string  
    prev_number1 = int(number1)
    prev_number2 = int(number2)
    
    number1 = np.round(number1 / divider).astype('int')
    number2 = np.round(number2 / divider).astype('int')
    
    # new_string = re.sub(r'(?<!\d)\d+(\.\d+)?(?!\d)', lambda match: str(number1) if float(match.group(0)) == number1 else str(number2), string)
    new_string = string.replace(str(prev_number1), str(number1)).replace(str(prev_number2), str(number2))
    return new_string

def extract_questions_and_answers(file_path, name, questions_and_answers):
    with open(file_path, 'r') as file:
        content = file.read()
    # name = file_path.split('/')[-1]
    max_frame = load_dict_from_file('vid_frames_breakfast.json')[name]
    sections = content.replace('Format 1:\n','').replace('Format 2:\n','').replace('Format 1: \n','').replace('Format 2: \n','').split('\n')
    cur_q = None
    for section in sections:
        # if 'cam01/P25_sandwich' in file_path:
        #     print(section)
        section = section.replace('A: ', '').replace('A:\n', '').replace('A:', '')
        if section.strip() == '': continue
        if 'Q:' in section:
            cur_q = section.replace('Q: ', '')
            cur_q = divide_numbers_if_starts_with(cur_q.replace('\n',''), max_frame = max_frame)
            questions_and_answers[cur_q] = ""
        else:
            questions_and_answers[cur_q] += divide_numbers_if_starts_with(section.replace('\t','').strip(), max_frame = max_frame) + '\n'

    return questions_and_answers

def load_valid(path):
    lis = []
    with open(path, 'r') as f:
        for line in f.readlines():
            lis.append('/home/waleed/ahmed/LLM/Breakfast/Breakfast' + line.replace('\n', '') + '.labels')
    return lis

def add_additional_questions(path):
    if not os.path.exists(path):
        path = path.replace('ch0', 'ch1')
    action = path.split('/')[-1].split('.')[0].split('_')[1]
    question = 'List down all the segments. action is ' + action
    answer = ""
    with open(path, 'r') as f:
        for line in f.readlines():
            answer += line
    return question, answer

def main():
    args = parse_args()
    input_folder = args.input_json_file
    output_json_file = args.output_json_file

    output_json_contents = []
    i = 0
    augs = ['','flip', 'bright', 'bright_flip','contrast', 'contrast_flip', 'crop', 'crop_flip', 'compress', 'compress_flip', 'sharp', 'sharp_flip']
    skip_them = load_valid('valid.txt')
    for file in tqdm.tqdm(sorted(glob.glob(input_folder + '*'))):
        for cam in glob.glob(file + '/*'):
            for action in glob.glob(cam + '/*'):
                video = action.split('/')[-1].split('.')[0] + '_' + cam.split('/')[-1]
                if action in skip_them:
                    # print('skipped', video)
                    continue
                # try:
                questions_and_answers = {}
                question, answer = add_additional_questions(action.replace('Breakfast/Breakfast/','Breakfast/Videos/BreakfastII_15fps_qvga_sync/'))
                questions_and_answers[question] = answer
                questions_and_answers = extract_questions_and_answers(action, video, questions_and_answers)
                # except Exception as e:
                #     print('Error on', action)
                #     print(e)
                #     break
                j = 0
                rep = 0
                for question, answer in questions_and_answers.items():
                    question = question.replace('\n','')
                    answer = answer.replace('_', ' ')
                    if j == 0: rep = 12 
                    else: rep = 2
                    for k in range(rep):
                        if not augs[k % len(augs)] == '':
                            new_video = "_".join(video.split('_')[:-1]) + '_' + augs[k % len(augs)] + '_' + video.split('_')[-1]
                        else:
                            new_video = video # uncomment all
                        # #new_video = video + '_all'
                       # new_video=video #new addtiion
                        output_content = {'id': new_video, 'video': f"{new_video}.pkl", 'conversations': []}
                        if i % 2 == 0:
                            output_content['conversations'].append({'from': 'human', 'value': f"{question}\n<video>"})
                        else:
                            output_content['conversations'].append({'from': 'human', 'value': f"<video>\n{question}"})
                        output_content['conversations'].append({'from': 'gpt', 'value': answer})
                        i += 1
                        output_json_contents.append(output_content)
                    j += 1
    
    print(f"Total annotations retained: {len(output_json_contents)}")
    with open(output_json_file, 'w') as f:
        json.dump(output_json_contents, f)


if __name__ == "__main__":
    main()