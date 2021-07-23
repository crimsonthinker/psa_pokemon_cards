import argparse
from trainers.vgg16_pokemon_grader import VGG16PokemonGrader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        default='vgg16_pokemon_grader', 
        nargs='?',
        help="Learning rate for the model")
    parser.add_argument(
        "--model_datetime", 
        type=str, 
        default=None, 
        nargs='+',
        help="Timestamp of the model. Default as the newest")
    parser.add_argument(
        "--img_dir", 
        type=str,
        help = "Image directory for prediction")
    parser.add_argument(
        "--dest_pred_dir", 
        type = str,
        default = None,
        help = 'Image destination path for csv'
    )
    args = parser.parse_args()

    if args.model == 'simple_image_classifier':
        classifier = VGG16PokemonGrader()

    # load the model from .checkpoints
    classifier.load(args.model_datetime)

    # predict from the image directory
    predictions = classifier.predict(
        args.img_dir,
        args.dest_pred_dir
    )