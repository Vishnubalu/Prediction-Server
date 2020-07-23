from django.contrib import admin
from django.urls import path
from predictionApi import views

urlpatterns = [
    path('', views.react),
    path('symptoms/', views.liverDiseaseSymptoms),
    path('predictdisease/', views.liverDiseasePrediction),
    path('download/', views.downloadsample),
    path('result/', views.downloadResult),
    path('upload/', views.uploadFile),
    path('admin/', admin.site.urls),
]
