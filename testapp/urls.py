from django.urls import path
from testapp import views

urlpatterns = [
    path('', views.testapp, name='testapp'),
    path('log_out', views.log_out, name='log_out'),
    path('register', views.Link1, name='Link1'),
    path('Link11', views.Link11, name='Link11'),
    path('userlogin', views.userlogin, name='userlogin'),
]
