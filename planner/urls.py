from django.urls import path
from . import views

urlpatterns = [
    path('', views.planner_view, name='planner_home'),
]