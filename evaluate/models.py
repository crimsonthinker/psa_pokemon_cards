from django.db import models

# Create your models here.
class FrontImage(models.Model):
    front_img = models.ImageField(upload_to = 'images/')

class BackImage(models.Model):
    back_img = models.ImageField(upload_to = 'images/')