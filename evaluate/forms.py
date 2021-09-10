from django import forms

from . import models

class FrontImageForm(forms.ModelForm):

    class Meta:
        model = models.FrontImage
        fields = ['front_img']

class BackImageForm(forms.ModelForm):

    class Meta:
        model = models.BackImage
        fields = ['back_img']