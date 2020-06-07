from django.db import models


class Post(models.Model):
    name = models.CharField(max_length=500)
    image = models.ImageField(blank=True)
    result = models.ImageField(blank=True)