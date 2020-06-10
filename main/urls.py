from django.urls import path
from django.conf.urls import url, include
from django.conf.urls.static import static
from django.conf import settings
from rest_framework import routers
from . import views


router = routers.DefaultRouter()
router.register(r'posts', views.PostViewSet)

urlpatterns = [
    path('', views.analysis_view, name="analysis"),
    path('test/', views.test, name="test"),
    url(r'^', include(router.urls)),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework')),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
