from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from main.models import Post
import analysis

def analysis_view(request):

    data = {
        "message" : "message",
        "image" : ""
    }
    # json_dumps_params = {"ensure_ascii":True} //문자 아스키코드로 바꿔서 전달할때 추가하기
    return JsonResponse(data)


def test(request):

    if request.method == "POST":
        
        post = Post()
        post.name = request.FILES["image"].name 
        post.image = request.FILES["image"]
        post.save()
        
        analysis.draw_line(post.pk)

        data = {
            "message" : "message",
            "image" : post.image.url 
        }
        return JsonResponse(data)
    else:
        return render(request, "test.html")
