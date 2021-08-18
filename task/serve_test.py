import argparse
import pandas as pd
from models.vgg16_grader import VGG16Grader
import numpy as np
from utils.preprocessor import psa_score

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
    parser.add_argument("--grade_ground_truth_path", type = str,default = None,
        help = 'Path of ground truth grade file.'
    )
    args = parser.parse_args()

    if len(args.model_score_type) == 0:
        score_types = ['Centering', 'Surface', 'Corners', 'Edges']
    else:
        score_types = args.model_score_type

    series = []
    for score_type in score_types:
        if args.model == 'vgg16_grader':
            classifier = VGG16Grader(grade_name = score_type)

        # load the model from checkpoint
        classifier.load(args.model_datetime)

        # predict from the image directory
        predictions_df = classifier.predict(args.img_dir)

        series.append(predictions_df)

    final = pd.concat(series, axis = 1)
    final.index = final.index.astype(np.int64)
    final['Grade'] = psa_score(final.mean(axis = 1).to_numpy())

    # merge with true grade
    if args.grade_ground_truth_path is not None:
        true_grade = pd.read_csv(args.grade_ground_truth_path, index_col = 'Identifier')
        true_grade.index = true_grade.index.astype(np.int64)

        # rename columns
        e = "Ground_Truth"
        columns =  final.columns
        ground_truth_name = {k : f"{k}_{e}" for k in columns}
        true_grade = true_grade.rename(columns = ground_truth_name)

        final = final.merge(true_grade, on = 'Identifier')
        new_sorted_columns = []
        for column in columns:
            new_sorted_columns.append(column)
            new_sorted_columns.append(f"{column}_{e}")
        final = final[new_sorted_columns]
    final.to_csv("result.csv", index_label = 'Identifier')
        