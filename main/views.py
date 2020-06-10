from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from main.models import Post
import analysis

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
        return JsonResponse(data)
    else:
        return HttpResponse("worng request")



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
