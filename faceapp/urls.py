from django.urls import path
from . import views

urlpatterns = [
    path('face-register/', views.face_register, name='face_register'),
    path('mark-attendance/', views.mark_attendance, name='mark_attendance'),
    # Removed get-new-captcha endpoint
]