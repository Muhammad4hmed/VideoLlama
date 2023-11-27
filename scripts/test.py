import os
import glob

# Specify the folder path where you want to count .pkl files
folder_path = '../features_breakfast_fawad_final-v2'

# Use glob to find all .pkl files in the folder
pkl_files = glob.glob(os.path.join(folder_path, '*.pkl'))

# Get the count of .pkl files
count = len(pkl_files)

print(f"Number of .pkl files in the folder: {count}")
