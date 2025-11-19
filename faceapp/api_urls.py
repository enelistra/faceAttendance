"""
API URL configuration for faceapp
"""

from django.urls import path
from . import views

urlpatterns = [
    # Face registration API endpoint
    path('face-register/', views.face_register, name='api_face_register'),
    
    # Attendance marking API endpoint  
    path('mark-attendance/', views.mark_attendance, name='api_mark_attendance'),
    
    # External employee mapping API endpoint
    path('map-employee/', views.api_map_employee, name='api_map_employee'),
]