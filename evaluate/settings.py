from pathlib import Path
import os

from task.loaders import VGG16PreProcessor
from models.vgg16_grader import VGG16GraderCentering
from models.vgg16_grader import VGG16GraderCorners
from models.vgg16_grader import VGG16GraderEdges
from models.vgg16_grader import VGG16GraderSurface

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# global grader models for API
CROPPER = VGG16PreProcessor(512,512,3)

ASPECTS = ['Centering', 'Surface', 'Corners', 'Edges']
class_mapper = {
    'Centering' : VGG16GraderCentering,
    'Surface' : VGG16GraderSurface,
    'Corners' : VGG16GraderCorners,
    'Edges' : VGG16GraderEdges
}

GRADERS = {}
for score_type in ASPECTS:
    GRADERS[score_type] = class_mapper[score_type](
        max_score = 10.0,
        img_height = 512,
        img_width = 512,
        dim = 4, # default as 4 dim
        clean_log = False,
        clean_checkpoints = False
    )
    GRADERS[score_type].load()

