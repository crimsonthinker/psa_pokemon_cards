# PSA Cards Grading Module

## Introduction

PSA Cards Grading Module is an AI model built for grading the quality of trading cards of 4 main aspects: Centering, Corners, Edges, and Surface. It uses one of the instances of transfer learning technique, fine-tuning, to construct the model using baseline models such as VGG16, ResNet, or MobileNet.

The current model, built on VGG16, provides two seperate network flow to learn different aspects of the image. Then, all information is gathered as a single 1-D layer, and it is propagated through several layers before outputing a grade for an aspect. Before feeding the images to the model, different layers of preprocessing are adopted (inclcuding cropping the card content using U-Net) to transform the images into a more "learnable" format. 

Based on previous evaluation, it can be seen that there are some very substantial results as the average difference between the target grade and the output from the model is about 0.5 (on the scale of 10).

## Project structure
Before diving in the project, we will describe the current structure of this project, along with some important folders that you should notice.
```
- .log: Containing logs of the most recent training session
- analysis: jupyter notebooks of analysis of grading cards and the model's result. To see how grading models have performed, prefer to the "analysis/validation_analysis/result_visualization.ipynb".
- checkpoint: checkpoints of the models. For the model that assists in cropping card, it is stored in "checkpoint/cropper". As for the grading models, it is in "checkpoint/vgg16_grader_M" (with M as the score aspect).
- data: Containing images of cards. This is the folder that is used for storing card image for training the grading models. The images are stored in "data/[id]/[back and front].jpg (with id is the id of the card). Furthermore, this folder also contains a grades.csv, which is a list of scores of each aspect or the cards in the folder data.
- models: Model class.
- task: A list of available tasks written in Python.
- unet_labeled: Containing images of cards, together with labels for the cropping model (U-Net). Images are stored in "unet_labeled/[test or train folder]/[id]/[back and front].jpg, and the annotations (generated by Labelme annotation tool) are stored in "unet_labeled/[test or train]/[id]/[back and front].json.
- utils: Other utilities function written in Python.
```

## Updating the model

We list of several existing tasks to retrain the model with new images. Currently, there are 4 main tasks:
- Extracting preprocessed data for cropper model.
- Extracting preprocessed data for grading models.
- Training cropping model from preprocessed data.
- Training grading models from preprocessed data.

The commands for each of these tasks are as follows:

```
To preprocess data for cropper model
python3 -m task.extract_cropper_data \
    --origin_img_height [defaults as 3147] \ # Origin height of the image
    --origin_img_width [defaults as 1860]\ # Origin width of the image
    --dim [defaults as 3] # dimension of the image
Data will be saved in preprocessed_data/UNET
```

```
To preprocess data for grading models
python3 -m task.extract_grade_data \
    --train_directory [train_directory] \ # directory containing card images.
    --origin_img_height [int] \ # height of the original trading card image.
    --origin_img_width [int] \ # width of the original trading card image.
    --model_score_type [str] \ #score type to extract data.
    --enable_ray # choosing multiprocessing approach to preprocess data 
Data will be saved in preprocessed_data/[score_type]
```

```
To train cropping model
python3 -m task.train_cropper \
    --origin_img_height [int] \ # original height of the image
    --origin_img_width [int] \ # original width of the image
    --img_height [int] \ # output image height of the model
    --img_width [int] \ # output image width of the model
    --dim [int] \ # dimension of the image
    --epochs [int] \ # number of training rounds
    --batch_size [int] \ # batch size
    --val_ratio [float] # ratio of the validation dataset
```

```
python3 -m task.train_grader \
    --clean_log \ # enable this if log data needs to be cleaned
    --clean_checkpoints \ # enable this if model checkpoints need to be cleaned
    --img_height [img_height] \ # image height for model construction. Default as 512
    --img_width [img_width] \ # image width for model construction. Default as 512
    --batch_size [batch_size] \ # batch size for dataset
    --epochs [epochs] \ # Number of training rounds for the model.
    --val_ratio [float] \ # ratio of the validation dataset
    --learning_rate [learning_rate] \ # learning rate for the model. Default as 0.001
    --model_score_type [list of score types] \ # List of score types. Currently it supports Centering, Surface, Edges, and Corners
```

## Installation and usage

Below are the steps required to install and run the grading components:

- Install Miniconda based on your OS (https://docs.conda.io/en/latest/miniconda.html).

- From the project's root folder, create a Conda environment (the default name for this environment is 'psa'):
```
conda env create -f environment.yml
```
- Activate the Conda environment:
```
conda activate psa
```
- Make sure that the checkpoints are available in folder "checkpoint" (which includes "cropper", "vgg16_grader_Centering", "vgg16_grader_Corners", "vgg16_grader_Edges", and "vgg16_grader_Surface").
- Run this command to start Django web server:
```
python3 manage.py runserver
```
- From then, you can go to [this page](http://localhost:8000/evaluate/upload) to upload your image and get graded by the AI models.
