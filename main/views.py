from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from main.models import Post
import analysis
from rest_framework import viewsets
from main.serializers import PostSerializer
from django.views.decorators.csrf import csrf_exempt
import json
import base64
from django.utils import timezone, dateformat
from django.core.files.base import ContentFile

class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all().order_by('-pk')
    serializer_class = PostSerializer


@csrf_exempt
def analysis_view(request):

    if request.method == "POST":
        

        body_unicode = request.body.decode('utf-8')
        body = json.loads(request.body)
        print(request.body)
        content = body['b64_image']

        filename = str(dateformat.format(timezone.now(), 'Y-m-d_H-i-s')) + '.jpg'

        post = Post()
        post.image = ContentFile(base64.b64decode(content), name=filename)
        post.name = filename
        post.save()
        object, result_data, total_percent = analysis.draw_line(post.pk)

        user_image = base64.b64encode(post.image.read())
        result_image = base64.b64encode(object.result.read())

        print(test)

        data = {
            "message" : str(result_data[0]),            
            "image_name" : post.image.name.split('.')[0],
            "image" : str(user_image),
            "result" : str(result_image)
        }
        
        return JsonResponse(data)
    else:
        return HttpResponse("get is worng request")



def test(request):

    if request.method == "POST":
        
        post = Post()
        post.name = request.FILES["image"].name 
        post.image = request.FILES["image"]
        post.save()
        
        

        object, result_data = analysis.draw_line(post.pk)

        data = {
            "message" : result_data,
            "image" : post.image.url,
            "result" : object.result.url
        }
        return render(request, "test.html", data)
    else:
        return render(request, "test.html")
