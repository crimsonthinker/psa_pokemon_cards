import argparse
from trainers.vgg16_grader import VGG16Grader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='vgg16_grader', nargs='?',
        help="Model name for classification")
    parser.add_argument("--model_score_type", type=str, default=[], nargs='+',
        help="Score type of the model. Leave blank if run all.")
    parser.add_argument("--model_datetime", type=str, default=None, nargs='?',
        help="Timestamp of the model. Default as the newest")
    parser.add_argument("--img_dir", type=str, default=None, nargs='?',
        help = "Image directory for prediction")
    parser.add_argument("--dest_pred_dir", type = str,default = None,
        help = 'Image destination path for csv'
    )
    args = parser.parse_args()

    if len(args.model_score_type) == 0:
        score_types = ['Centering', 'Surface', 'Corners', 'Edges']
    else:
        score_types = args.model_score_type

    for score_type in score_types:
        if args.model == 'vgg16_grader':
            classifier = VGG16Grader(grade_name = score_type)

        # load the model from .checkpoints
        classifier.load(args.model_datetime)

        # predict from the image directory
        predictions = classifier.predict(
            args.img_dir,
            args.dest_pred_dir
        )