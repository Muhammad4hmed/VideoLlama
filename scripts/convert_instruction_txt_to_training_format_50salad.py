import json
import argparse
import glob
import json
import numpy as np
import re

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
    divider = np.round(max_frame/picked_frames)
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

def extract_questions_and_answers(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    name = file_path.split('/')[-1]
    max_frame = load_dict_from_file('vid_frames.json')[name]
    sections = content.replace('Format 1:\n','').replace('Format 2:\n','').split('\n')
    questions_and_answers = {}
    cur_q = None
    for section in sections:
        section = section.replace('A: ', '').replace('A:\n', '').replace('A:', '')
        if section.strip() == '': continue
        if 'Q:' in section:
            cur_q = section.replace('Q: ', '')
            cur_q = divide_numbers_if_starts_with(cur_q.replace('\n',''), max_frame = max_frame)
            questions_and_answers[cur_q] = ""
        else:
            questions_and_answers[cur_q] += divide_numbers_if_starts_with(section.replace('\t','').strip(), max_frame = max_frame) + '\n'

    return questions_and_answers

def main():
    args = parse_args()
    input_folder = args.input_json_file
    output_json_file = args.output_json_file

    output_json_contents = []
    i = 0
    skip_them = [
        'rgb-01-2',
        'rgb-03-2',
        'rgb-05-2',
        'rgb-08-2',
        'rgb-10-2',
        'rgb-27-2',
        'rgb-15-2',
        'rgb-18-2',
        'rgb-24-2',
        'rgb-11-2',
    ]
    for file in glob.glob(input_folder + '*'):
        questions_and_answers = extract_questions_and_answers(file)
        video = file.split('/')[-1]
        if video in skip_them:
            print('skipped', video)
            continue
        j = 0
        rep = 0
        for question, answer in questions_and_answers.items():
            question = question.replace('\n','')
            answer = answer.replace('_', ' ')
            if j == 0: rep = 150
            else: rep = 90
            for k in range(rep):
                output_content = {'id': video, 'video': f"{video}.pkl", 'conversations': []}
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
