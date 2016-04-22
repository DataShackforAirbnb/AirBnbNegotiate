from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from .models import Search, User, Availability, Listing
from django.core.urlresolvers import reverse
import datetime
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
import logging
from django.template import RequestContext
from django.shortcuts import render_to_response



discount_threshold = 0.8


logger = logging.getLogger(__name__)
 
def myfunction():
    logger.debug("this is a debug message!")
 
def myotherfunction():
    logger.error("this is an error message!!")

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

def filter_listings(request):
    search = Search()
    context = RequestContext(request)
    if request.method == 'GET':
        filter_room = request.GET.getlist('filter_room[]' )
        filter_location = request.GET.getlist('filter_location[]' )        
        lower_price= request.GET.get('lower_price' )
        upper_price= request.GET.get('upper_price' )
    listings = []    
    destination = ('New York NY, United States')
    checkin_date = datetime.datetime.strptime(request.GET['date_in'], '%Y-%m-%d').strftime('%Y-%m-%d')
    checkout_date =datetime.datetime.strptime(request.GET['date_out'], '%Y-%m-%d').strftime('%Y-%m-%d')
    search.checkin_date = checkin_date
    search.chechout_date = checkout_date
    search.destination = destination
    listings = Availability.objects.filter(start_date = checkin_date, end_date = checkout_date, 
    avg_price__lte=upper_price,avg_price__gte=lower_price)    
    if len(filter_room)>0:
        listings = listings.filter(property_id__room_type__in = filter_room)
    if len(filter_location)>0:
        listings = listings.filter(property_id__neighbourhood__in = filter_location)        
    print listings    
    search.save()    
    return render_to_response('Negot/result_list.html', {'listings': listings }, context)


def auth_view(request):
    email = request.POST.get('username', 'qing')
    password = request.POST.get('password', 'tb314630')
    user = authenticate(username = email, password = password)

    print 'email:', email, 'password:', password
    if user is not None:
        if user.is_active:
            login(request, user)
            # Redirect to a success page.
            return render(request, 'Negot/index.html', {'user': user})
        else:
            return render(request, 'Negot/index.html', {'errors': 'Your account is inactive, please activate it before logging in.'})
            # Return a 'disabled account' error message
    else:
        # Return an 'invalid login' error message.
        return render(request, 'Negot/index.html', {'errors': 'Your username and password didn\'t match. Please try again.'})

@login_required
def log_out(request):
    logout(request)
    return render(request, 'Negot/index.html', {})