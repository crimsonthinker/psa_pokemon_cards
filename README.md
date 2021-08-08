# PSA Grading Cards

## Introduction

PSA Grading cards is an AI model built for scoring the quality of trading cards on 4 main aspects: Centering, Corners, Edges, and Surface. It uses one approach of transfer learning technique, fine-tuning, to construct the model using baseline models such as VGG16, ResNet, MobileNet.

The current model, built on VGG16, provided a very substantial result as the average difference between the target grade and the output from the model is about 0.1 (on the scale of 10).

## Usage

The command use for training follows:

```
python3 -m task.train \
    --model [model_name] \ #Default as vgg16_grader
    --skip_preprocessing \ #enable this if data preprocessing can be skipped
    --clean_log \ # enable this if log data needs to be cleaned
    --clean_checkpoints \ # enable this if model checkpoints need to be cleaned
    --train_directory \ # original train directory
    --img_height [img_height] \ # image height for model construction. Default as 256
    --img_width [img_width] \ # image width for model construction. Default as 256
    --dim [dim] \ # dimension of the image. Default as 3 (representing RGB color)
    --batch_size [batch_size] \ # batch size for dataset
    --epochs [epochs] \ # Number of training rounds for the model
    --learning_rate [learning_rate] \ # learning rate for the model. Default as 0.001
    --model_score_type [list of score types] \ # List of score types. Currently it supports Centering, Surface, Edges, and Corners
```

An example of model grading module is also provided in task/server_test.py. Command usage is as follows:
```
python3 -m task.serve_test \
    --model [model_name] \ #Default as vgg16_grader
    --model_score_type [list of score types] \ # List of score types. Currently it supports Centering, Surface, Edges, and Corners
    --model_datetime [datetime] \ #Datetime of the checkpoint. Default as the newest
    --img_dir [img_dir] # image directory for grading
    --grade_ground_truth_path [file path] #path of ground truth grades
```
