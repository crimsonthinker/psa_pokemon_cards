import argparse
from utils.loader import ImageLoader
from trainers.vgg16_pokemon_grader import VGG16PokemonGrader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='simple_image_classifier', nargs='?',
        help="Model name for temple classification")
    parser.add_argument("--skip_reloading", action='store_true', default = False,
        help="Skip reloading images into default preprocessed image folder")
    parser.add_argument("--clean_log", action='store_true', default = False,
        help="Clean log folder")
    parser.add_argument("--clean_checkpoints", action='store_true', default = False,
        help="Clean checkpoints folder of the model")
    parser.add_argument("--save_evaluation", action='store_true', default = False,
        help="Save visualization result in .log folder")
    parser.add_argument("--visualize_result", action='store_true', default = False,
        help="Visualize the result")
    parser.add_argument("--train_directory", type=str, default='temples-train-hard/train', nargs='?',
        help="Training directory")
    parser.add_argument("--img_height", type=int, default=256, nargs='?',
        help="Image height for the training session")
    parser.add_argument("--img_width", type=int, default=256, nargs='?',
        help="Image width for the training session")
    parser.add_argument("--batch_size", type=int, default=32, nargs='?',
        help="Batch size for training session")
    parser.add_argument("--epochs", type=int, default=2, nargs='?',
        help="Number of epochs for training session")
    parser.add_argument("--learning_rate", type=float, default=0.01, nargs='?',
        help="Learning rate for the model")
    args = parser.parse_args()

    # runt raining session
    image_dataset = ImageLoader(
        skip_reloading = args.skip_reloading,
        train_directory = args.train_directory,
        img_height = args.img_height,
        img_width = args.img_width,
        batch_size = args.batch_size
    )
    # load the iamge from train directory
    image_dataset.load()

    if args.model == 'vgg16_pokemon_grader':
        model = VGG16PokemonGrader(
            class_names = image_dataset.class_names,
            img_height = args.img_height,
            img_width = args.img_width,
            learning_rate = args.learning_rate,
            epochs = args.epochs,
            clean_log = args.clean_log,
            clean_checkpoints = args.clean_checkpoints
        )

    model.train_and_evaluate(image_dataset)

    # visualize results
    if args.visualize_result or args.save_evaluation:
        model.visualize_evaluation(
            args.visualize_result,
            args.save_evaluation)
    
    # save the model
    model.save()
