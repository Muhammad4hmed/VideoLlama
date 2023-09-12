import tqdm
import cv2
import numpy as np
import albumentations as A
import argparse
import glob

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=False, default = "/home/waleed/ahmed/LLM/Breakfast/Videos/BreakfastII_15fps_qvga_sync/")
    parser.add_argument('--fps', help='FPS', required=False, default = 15)
    
    return parser.parse_args()

def get_augmentations(image):
    transforms = ([
        A.Compose([
            A.HorizontalFlip(p=1.0)
        ]),
        A.Compose([
            A.RandomBrightnessContrast(p=1.0,brightness_limit = [0.2, 0.2],  contrast_limit = 0.0)
        ]),
        A.Compose([
            A.RandomBrightnessContrast(p=1.0,brightness_limit = [0.2, 0.2],  contrast_limit = 0.0),
            A.HorizontalFlip(p=1.0)
        ]),
        A.Compose([
            A.RandomBrightnessContrast(p=1.0,brightness_limit = 0.0,  contrast_limit = [-0.3, -0.3])
        ]),
        A.Compose([
            A.RandomBrightnessContrast(p=1.0,brightness_limit = 0.0,  contrast_limit = [-0.3, -0.3]),
            A.HorizontalFlip(p=1.0)
        ]),
        A.Compose([
            A.Crop(x_min=0, y_min=50, x_max = image.shape[1], y_max = image.shape[0], p = 1.0)
        ]),
        A.Compose([
            A.Crop(x_min=0, y_min=50, x_max = image.shape[1], y_max = image.shape[0], p = 1.0),
            A.HorizontalFlip(p=1.0)
        ]),
        A.Compose([
            A.ImageCompression(p=1.0, quality_lower=5, quality_upper=5, compression_type = A.ImageCompression.ImageCompressionType.WEBP)
        ]),
        A.Compose([
            A.ImageCompression(p=1.0, quality_lower=5, quality_upper=5, compression_type = A.ImageCompression.ImageCompressionType.WEBP),
            A.HorizontalFlip(p=1.0)
        ]),
        A.Compose([
            A.Sharpen(p = 1.0, alpha =(0.2, 0.3))
        ]),
        A.Compose([
            A.Sharpen(p = 1.0, alpha =(0.2, 0.3)),
            A.HorizontalFlip(p=1.0)
        ]),
    ], ['flip', 'bright', 'bright_flip','contrast', 'contrast_flip', 'crop', 'crop_flip', 'compress', 'compress_flip', 'sharp', 'sharp_flip'] )
    return transforms[0], transforms[1]

def load_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    images = []
    while success:
        images.append(image)
        success,image = vidcap.read()
        count += 1
    return np.array(images)

def save_video(path, aug_images, aug_name, fps):
    size = aug_images.shape[1], aug_images.shape[2], 3
    save_path = "/".join(path.split('/')[:-1]) + '/' + path.split('/')[-1].replace('.', f'_{aug_name}.')
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (size[1], size[0]))
    for i in range(aug_images.shape[0]):
        data = np.random.randint(0, 256, size, dtype='uint8')
        data[:] = aug_images[i]
        out.write(data)
    out.release()

def augment_videos(args):
    for person in tqdm.tqdm(sorted(glob.glob(args.video_dir + '/*'))):
        for cam in glob.glob(person + '/*'):
            videos = glob.glob(cam + '/*.avi')
            for video_path in videos:
                video = load_video(video_path)
                augs, names = get_augmentations(video[0])
                for transform, aug_name in zip(augs, names):
                    aug_images = []
                    for image in video:
                        aug_images.append(transform(image = image)['image'])
                    aug_images = np.array(aug_images)
                    save_video(video_path, aug_images, aug_name, args.fps)
                

if __name__=="__main__":
    args = parse_args()
    augment_videos(args)