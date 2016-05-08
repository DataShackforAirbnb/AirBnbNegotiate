from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from .models import Search, Availability, Listing
from django.core.urlresolvers import reverse
import datetime
from django.contrib.auth import authenticate, login, logout, get_user
from django.contrib.auth.decorators import login_required
import logging
from django.template import RequestContext
from django.shortcuts import render_to_response
from django.contrib.auth.models import User
from ml_class import Negot_Model
from django.conf import settings
import pandas as pd
import datetime, random, sha
from django.shortcuts import render_to_response, redirect
from django.core.mail import send_mail
from .forms import RegistrationForm
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.core.context_processors import csrf
from django.views.decorators.csrf import csrf_exempt

discount_threshold = 0.8
proba_threshold = 0.3
test = Negot_Model(calendar_date="20160401")
# test.Train()
# test.Preprocess()

logger = logging.getLogger(__name__)

def myfunction():
    logger.debug("this is a debug message!")
 
def myotherfunction():
    logger.error("this is an error message!!")

def convert_rating(listing):
    rating = listing.review_scores_rating
    output = [0,0,0,0,0]
    rating = round(rating/20.0 * 2.0) / 2.0
    if rating > 5 or rating <0:
        listing.review_scores_rating = output
        return listing

    output[0:int(rating)] = [1]*int(rating)
    if rating <5:
        output[int(rating)] = rating-int(rating)

    listing.review_scores_rating = output
    print listing.review_scores_rating
    return listing

def index(request):
    args = {}
    args.update(csrf(request))
    args['form'] = RegistrationForm()
    return render(request, 'Negot/index.html', {'args' : args})

def about(request):
    return render(request, 'Negot/about.html')

def results(request):

    return render(request, 'Negot/results.html')

def search(request):
    args = {}
    args.update(csrf(request))
    args['form'] = RegistrationForm()

    search = Search()
    try:
        checkin_date = datetime.datetime.strptime(request.POST['check-in'], '%b %d, %Y').strftime('%Y-%m-%d')
        checkout_date = datetime.datetime.strptime(request.POST['check-out'], '%b %d, %Y').strftime('%Y-%m-%d')
        destination = request.POST.get('destination', 'New York NY, United States')
        search.checkin_date = checkin_date
        search.chechout_date = checkout_date
        search.destination = destination

        #machine learning model integration
        ml_result = test.Predict(checkin_date, checkout_date)
        ml_result = [i for i in ml_result if i[2] > proba_threshold]
        if len(ml_result) > 50:
            ml_result = sorted(ml_result, key = lambda x: (- x[2] * [3]))[:50]
        result_ids = [i[0] for i in ml_result]
        result_listings = Listing.objects.filter(airBnbId__in = result_ids)

        ################################################
        # Sort the result according to
        # 1. discount * accept_proba
        # 2. number of reviews
        # 3. ratings
        ################################################

        ml_result= zip(result_listings, ml_result)
        ml_result = sorted(ml_result, key = lambda x: (- x[0].number_of_reviews, - x[0].review_scores_rating))

        # Record user and results to search history
        search.num_of_results = len(ml_result)
        if request.user.is_authenticated():
            search.user = get_user(request)

        #filter only the useful neighbourhoods
        # join_id= ml_result['listings'].values_list('airBnbId', flat=True)[:10]
        # neighbourhoods = Listing.objects.filter(airBnbId__in = join_id).values_list('neighbourhood', flat=True)
        neighbourhoods = result_listings.values_list('neighbourhood', flat=True)
        neighbourhoods= list(set(neighbourhoods))
        neighbourhoods= filter(lambda a: "nan" not in a, neighbourhoods)           
        # Apply discount threshold
    except (KeyError, Search.DoesNotExist):
        # Redisplay the index form with error Infomation.
        return render(request, 'Negot/index.html')
    else:
        search.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        listings = [i[0] for i in ml_result]

        #convert listing review scores to a length-5 tuple: 3.5->(1,1,1,0.5,0)
        listings = [convert_rating(i) for i in listings]

        orig_percent_off = [int(i[1][1] * 100) for i in ml_result]
        negot_proba = [int(i[1][2] * 100) for i in ml_result]
        discounted_price = [int((1-i[1][3]) * i[0].price) for i in ml_result]

        ml_result = zip(listings, orig_percent_off, negot_proba, discounted_price)
        return render(request, 'Negot/results.html', {'results': ml_result, 'part_results': ml_result[:5], 'neighbourhoods':neighbourhoods,
                                                      'checkin_date': checkin_date, 'checkout_date': checkout_date, 'args' : args})

def filter_listings(request):
    pass
def filter_maps(request):
    pass

def auth_view(request):
    email = request.POST.get('email', 'qing')
    password = request.POST.get('password')
    user = authenticate(username = email, password = password)

    print 'email:', email, 'password:', password
    if user is not None:
        if user.is_active:
            login(request, user)
            # Redirect to a success page.
            return render(request, 'Negot/index.html', {'user': user})
        else:
            return render(request, 'Negot/index.html', {'info': 'Your account is inactive, please activate it before logging in.'})
            # Return a 'disabled account' error message
    else:
        # Return an 'invalid login' error message.
        args = {}
        args.update(csrf(request))
        form = RegistrationForm()
        args['form'] = form

        return render(request, 'Negot/index.html', {'info': 'Your username and password didn\'t match. Please try again.','args' : args})

@login_required
def log_out(request):
    logout(request)

    args = {}
    args.update(csrf(request))
    form = RegistrationForm()
    args['form'] = form
    return render(request, 'Negot/index.html', {'args' : args})
    # return render(request, 'Negot/index.html')

@login_required
def track_click(request):
    pass

def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            return render(request, 'Negot/index.html', {'info': 'Register success! Please log in~'})
        else:
            args = {}
            args.update(csrf(request))
            args['form'] = form
            return render(request, 'Negot/index.html', {'info': 'Register Fail', 'args' : args})

    args = {}
    args.update(csrf(request))
    form = RegistrationForm()
    args['form'] = form
    return render(request, 'Negot/index.html', {'args' : args})
