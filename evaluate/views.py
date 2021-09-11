from evaluate.models import BackImage, FrontImage
from evaluate import forms
from django.http import HttpResponse
from django.shortcuts import render, redirect

from . import forms
from . import settings

import os
import cv2
from matplotlib.image import imread
import numpy as np

def image_view(request, evaluate = False):
    if request.method == 'POST':
        front_form = forms.FrontImageForm(request.POST, request.FILES)
        back_form = forms.BackImageForm(request.POST, request.FILES)


        if front_form.is_valid() and back_form.is_valid():
            front_form.save()
            back_form.save()

            # get images
            front_image_path = os.path.join(settings.BASE_DIR, FrontImage.objects.last().front_img.url[1:])
            back_image_path = os.path.join(settings.BASE_DIR, BackImage.objects.last().back_img.url[1:])

            # read images to numpy
            front_image = cv2.cvtColor(np.array(imread(front_image_path)), cv2.COLOR_BGR2RGB)
            back_image = cv2.cvtColor(np.array(imread(back_image_path)), cv2.COLOR_BGR2RGB)

            cropped_front_image = settings.CROPPER.crop(front_image)
            cropped_back_image = settings.CROPPER.crop(back_image)

            # final scores for each aspect
            scores = {}

            # Crop content
            for score_type in settings.ASPECTS:
                # append the preprocessed image
                preprocessed_image, residual = settings.CROPPER.preprocess(cropped_front_image, cropped_back_image, score_type)
                if preprocessed_image is not None:
                    resized_preprocessed_image = cv2.resize(preprocessed_image, (512, 512), cv2.INTER_AREA)
                    if residual is not None:
                        # concat the residual to the image
                        residual = cv2.resize(residual, (512, 512), cv2.INTER_AREA)
                        residual = np.expand_dims(residual, -1)
                        resized_preprocessed_image = np.concatenate([resized_preprocessed_image,residual], axis = 2)
                    scores[score_type] = round(settings.GRADERS[score_type].predict(resized_preprocessed_image)[0], 2)
                else:
                    # Error here
                    scores[score_type] = -1.0

            return render(
                request, "main.html", 
                {
                    'front_form' : front_form, 
                    'back_form' : back_form,
                    'front_path' : FrontImage.objects.last().front_img.url,
                    'back_path' : BackImage.objects.last().back_img.url,
                    'Centering' : scores['Centering'],
                    'Edges' : scores['Edges'],
                    'Corners' : scores['Corners'],
                    'Surface' : scores['Surface']
                })
    else:
        front_form = forms.FrontImageForm()
        back_form = forms.BackImageForm()

    return render(request, "main.html", {'front_form' : front_form, 'back_form' : back_form})

def success(request):
    return HttpResponse('successfully uploaded')