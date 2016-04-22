from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^about/', views.about, name = 'about'),
    url(r'^results/', views.results, name = 'results'),
    url(r'^search/', views.search, name = 'search'),
    url(r'^auth_view', views.auth_view, name = 'auth_view'),
    url(r'^logout', views.log_out, name = 'logout'),
    url(r'^filter_listings/$', views.filter_listings, name='filter_listings'),
 
]