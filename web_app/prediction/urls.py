from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("reports/<str:filename>/", views.report_image, name="report_image"),
]
