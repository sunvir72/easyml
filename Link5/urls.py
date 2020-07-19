from django.urls import path
from Link5 import views

urlpatterns = [
    path('', views.Link5, name='Link5'),
    path('rowcol/', views.rowcol, name='rowcol'),
    path('prec/', views.prec, name='prec'),
    path('prec_/', views.prec_, name='prec_'),
    path('savemodel/<mno>', views.savemodel, name='savemodel'),
    path('smml/', views.smml, name='smml'),
    path('sm_test/', views.sm_test, name='sm_test'),
    path('delsm/<smid>', views.delsm, name='delsm'),
]
