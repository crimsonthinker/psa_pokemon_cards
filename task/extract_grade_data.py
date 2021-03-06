import argparse
import json
import os

from task.loaders import VGG16PreProcessor
from utils.utilities import get_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_directory", type=str, default='data', nargs='?',
        help="Training directory")
    parser.add_argument("--grade_path", type=str, default='data/grades.csv', nargs='?',
        help="Path to grade data")
    parser.add_argument("--origin_img_height", type=int, default=3147, nargs='?',
        help="Origin image height for the training session")
    parser.add_argument("--origin_img_width", type=int, default=1860, nargs='?',
        help="Original image width for the training session")
    parser.add_argument("--model_score_type", type=str, default=[], nargs='+',
        help="Score type of the model. Leave blank if run all.")
    parser.add_argument("--enable_ray", action='store_true', default = False,
        help="Enable ray multiprocessing")
    args = parser.parse_args()

    if os.path.isfile(os.path.join("preprocessed_data", "metadata.json")):
        with open(os.path.join("preprocessed_data", "metadata.json"), 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # run raining session
    image_dataset = VGG16PreProcessor(
        train_directory = args.train_directory,
        grade_path = args.grade_path,
        origin_img_height = args.origin_img_height,
        origin_img_width = args.origin_img_width,
        enable_ray = args.enable_ray)

    if len(args.model_score_type) == 0:
        score_types = ['Centering', 'Surface', 'Corners', 'Edges']
    else:
        score_types = args.model_score_type

    logger = get_logger("Grade data extractor")

    
    for score_type in score_types:
        logger.info(f"Generating data of score {score_type}")
        # load the image from train directory
        image_dataset.work(score_type)

        metadata[score_type] = {
            'img_height' : 224,
            'img_width' : 224,
            'failed_images_identifiers' : image_dataset.failed_images_identifiers
        }

    # save metadata
    with open(os.path.join("preprocessed_data", "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent = 4)
