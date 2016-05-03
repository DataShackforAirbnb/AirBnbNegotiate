import urllib2
import time
import json
import pandas as pd
import random
import numpy as np
start = time.time()

#lenght of the calendar you want to scrape
months= 4

#ids of hosts
listing_info = pd.read_csv('../data/listings/nyc_listings.csv')
ids = list(listing_info['id'])
random.shuffle(ids)

calendars = dict()
# failed_ids = pd.read_json('calendars/failed_ids.json')

def getCalendar(ids, months, output_file = '../data/calendars/nyc_cal_may.json', logfile = '../data/calendars/failed_ids_may.json'):
    counter = 1
    lost_ids = []
    for s in ids:
        hostURL= 'https://www.airbnb.it/api/v2/calendar_months?key=d306zoyjsyarp7ifhu67rjxn52tv0t20&currency=EUR&locale=en&listing_id='+str(s)+'&month=3&year=2016&count='+str(months)+'&_format=with_conditions'
        # print hostURL
        print counter
        req = urllib2.Request(hostURL
        #simulate a browser not to be blocked
        ,headers={'User-Agent' : "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/534.30 (KHTML, like Gecko) Ubuntu/11.04 Chromium/12.0.742.112 Chrome/12.0.742.112 Safari/534.30"})
        # response = urllib2.urlopen(req)
        # data = json.load(response)
        # calendars[s] = data
        # counter += 1
        try:
            response = urllib2.urlopen(req)
            data = json.load(response)
            calendars[s] = data
            counter += 1
        except:
            # ids.append(s)
            lost_ids.append(s)

            # print e.fp.read()
            time.sleep(15)
        time.sleep(random.uniform(0.5, 8))

        if(counter % 100 == 0):
            # with open('calendars/' + str(time.time())+'.json', 'w') as outfile:
            with open(output_file, 'w') as outfile:
                json.dump(calendars, outfile)
            with open(logfile, 'w') as outfile:
                json.dump(lost_ids, outfile)

    # failed_ids = pd.read_json(logfile)
    # while len(failed_ids) != 0:
    #     print 'scraping failed json, length: ',len(failed_ids)
    #     getCalendar(failed_ids, months, output_file = 'calendars/nyc_cal_' + str(len(failed_ids)) + '.json')
    return

getCalendar(ids, months)

print 'Total Execution Time:', time.time() - start