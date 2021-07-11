from django.urls import path
from . import views

urlpatterns = [
    path('stream/', views.video_feed, name="video_feed"),
    path('show/', views.show, name='show')
]

