import argparse
from utils.loader import ImageLoader
from models.vgg16_grader import VGG16Grader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='vgg16_grader', nargs='?',
        help="Model name for grading trading cards")
    parser.add_argument("--clean_log", action='store_true', default = False,
        help="Clean log folder")
    parser.add_argument("--clean_checkpoints", action='store_true', default = False,
        help="Clean checkpoints folder of the model")
    parser.add_argument("--train_directory", type=str, default='data', nargs='?',
        help="Training directory")
    parser.add_argument("--img_height", type=int, default=512, nargs='?',
        help="Image height for the training session")
    parser.add_argument("--img_width", type=int, default=512, nargs='?',
        help="Image width for the training session")
    parser.add_argument("--dim", type=int, default=3, nargs='?',
        help="Image didmension for the training session")
    parser.add_argument("--batch_size", type=int, default=8, nargs='?',
        help="Batch size for training session")
    parser.add_argument("--epochs", type=int, default=15, nargs='?',
        help="Number of epochs for training session")
    parser.add_argument("--learning_rate", type=float, default=0.001, nargs='?',
        help="Learning rate for the model")
    parser.add_argument("--model_score_type", type=str, default=[], nargs='+',
        help="Score type of the model. Leave blank if run all.")
    args = parser.parse_args()

    # run raining session
    image_dataset = ImageLoader(
        skip_preprocessing = True,
        train_directory = args.train_directory,
        img_height = args.img_height,
        img_width = args.img_width,
        batch_size = args.batch_size
    )

    if len(args.model_score_type) == 0:
        score_types = ['Centering', 'Surface', 'Corners', 'Edges']
    else:
        score_types = args.model_score_type

    for score_type in score_types:
        print(f"Training dataset on score {score_type}")
        # load the image from train directory
        image_dataset.load(score_type)

        if args.model == 'vgg16_grader':
            model = VGG16Grader(
                grade_name = score_type,
                max_score = image_dataset.max_score,
                img_height = args.img_height,
                img_width = args.img_width,
                dim = args.dim,
                learning_rate = args.learning_rate,
                epochs = args.epochs,
                clean_log = args.clean_log,
                clean_checkpoints = args.clean_checkpoints
            )
            model.load()

        # train the model
        # new model already been saved in this function
        model.train_and_evaluate(image_dataset)
        
        # save the model
        model.save_metadata()
