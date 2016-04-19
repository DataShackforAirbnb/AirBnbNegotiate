from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^about/', views.about, name = 'about'),
    url(r'^results/', views.results, name = 'results'),
    url(r'^search/', views.search, name = 'search'),
    url(r'^auth_viwe', views.auth_view, name = 'auth_view'),
    url(r'^logout', views.log_out, name = 'logout'),
]