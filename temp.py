import json 
import pickle 

def load_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# with open('features_breakfast_videochatgpt/P03_cereals_webcam02.pkl', "rb") as f:
#     features = pickle.load(f)

# print(features.shape)
labels = load_dict_from_file('video_chatgpt_training.json')
print(labels[13])

# import glob

# print(glob.glob('features_breakfast/*')[:5])