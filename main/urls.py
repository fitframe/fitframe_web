from django.urls import path
from . import views

urlpatterns = [
    path("", views.livefe, name="LiveFeed")
]