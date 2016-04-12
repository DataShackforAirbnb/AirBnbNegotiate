from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from .models import Search, User, Availability, Listing
from django.core.urlresolvers import reverse
import datetime

discount_threshold = 0.8
def index(request):
    return render(request, 'Negot/index.html')

def about(request):
    return render(request, 'Negot/about.html')

def results(request):
    return render(request, 'Negot/results.html')

def search(request):
    search = Search()
    try:
        checkin_date = datetime.datetime.strptime(request.POST['check-in'], '%b %d, %Y').strftime('%Y-%m-%d')
        checkout_date = datetime.datetime.strptime(request.POST['check-out'], '%b %d, %Y').strftime('%Y-%m-%d')
        destination = request.POST.get('destination', 'New York NY, United States')

        search.checkin_date = checkin_date
        search.chechout_date = checkout_date
        search.destination = destination

        results = Availability.objects.filter(start_date = checkin_date, end_date = checkout_date)

        # Apply discount threshold
    except (KeyError, Search.DoesNotExist):
        # Redisplay the index form with error Infomation.
        return render(request, 'Negot/index.html')
    else:
        search.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return render(request, 'Negot/results.html', {'listings': results, 'checkin_date': checkin_date, 'checkout_date': checkout_date})