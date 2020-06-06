from django.shortcuts import render
from django.http import HttpResponse, JsonResponse


def analysis_view(request):

    data = {
        "message" : "message",
        "image" : ""
    }
    # json_dumps_params = {"ensure_ascii":True} //문자 아스키코드로 바꿔서 전달할때 추가하기
    return JsonResponse(data)