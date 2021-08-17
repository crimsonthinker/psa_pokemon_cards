import argparse
from utils.loader import GraderImageLoader
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_directory", type=str, default='data', nargs='?',
        help="Training directory")
    parser.add_argument("--img_height", type=int, default=512, nargs='?',
        help="Image height for the training session")
    parser.add_argument("--img_width", type=int, default=512, nargs='?',
        help="Image width for the training session")
    parser.add_argument("--dim", type=int, default=3, nargs='?',
        help="Image didmension for the training session")
    parser.add_argument("--model_score_type", type=str, default=[], nargs='+',
        help="Score type of the model. Leave blank if run all.")
    parser.add_argument("--enable_ray", action='store_true', default = False,
        help="Enable ray multiprocessing")
    args = parser.parse_args()

    if os.path.isfile(os.path.join(".preprocessed_train", "metadata.json")):
        with open(os.path.join(".preprocessed_train", "metadata.json"), 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # run raining session
    image_dataset = GraderImageLoader(
        skip_preprocessing = False,
        train_directory = args.train_directory,
        img_height = args.img_height,
        img_width = args.img_width,
        enable_ray = args.enable_ray)

    if len(args.model_score_type) == 0:
        score_types = ['Centering', 'Surface', 'Corners', 'Edges']
    else:
        score_types = args.model_score_type

    
    for score_type in score_types:
        print(f"Generating data of score {score_type}")
        # load the image from train directory
        image_dataset.load(score_type)

        metadata[score_type] = {
            'img_height' : args.img_height,
            'img_width' : args.img_width,
            'dim' : 3,
            'failed_images_identifiers' : image_dataset.failed_images_identifiers
        }

    # save metadata
    with open(os.path.join(".preprocessed_train", "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent = 4)
