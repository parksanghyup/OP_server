from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from main.models import Post
import analysis
from rest_framework import viewsets
from main.serializers import PostSerializer
from django.views.decorators.csrf import csrf_exempt

class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all().order_by('-pk')
    serializer_class = PostSerializer


@csrf_exempt
def analysis_view(request):


    if request.method == "POST":
        
        post = Post()
        post.name = request.FILES["image"].name 
        post.image = request.FILES["image"]
        post.save()      
        object = analysis.draw_line(post.pk)

        data = {
            "message" : "message",
            "image" : post.image.url,
            "result" : object.result.url
        }
        return HttpResponse("HIHIHIHHI")
    else:
        return JsonResponse({"Worng":"WORNG"})



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
