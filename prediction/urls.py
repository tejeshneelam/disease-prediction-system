# disease_prediction_project/prediction/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('chatbot/', views.chatbot, name='chatbot'), # New URL for the chatbot
    path('contact/', views.contact, name='contact'),
    # ... (in prediction/urls.py) ...
    path('hospitals/', views.hospitals, name='hospitals'),
]