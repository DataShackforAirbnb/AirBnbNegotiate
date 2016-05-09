__author__ = 'fqian'
import pandas as pd
import numpy as np
import datetime as dt
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer

class Model(object):

    def __init__(self,
                 input_dir="input", 
                 intermediate_dir="intermediate",
                 output_dir="output",
                 calendar_date=dt.date.today().strftime("%Y%m%d"),
                 city="nyc"):
        self.city = city
        self.input_dir = input_dir
        self.intermediate_dir = intermediate_dir
        self.output_dir = output_dir

        # TODAY's DATE
        self.today_parsed = dt.datetime.today()
        self.calendar_date = calendar_date
        self.calendar_date_parsed = dt.datetime.strptime(calendar_date, "%Y%m%d")

        # Imputer if NAN
        self.imputer = Imputer(copy=True, missing_values='NaN', strategy='mean', axis=1)

    #################################################################################################################
    def Train(self):
        input_dir = self.input_dir
        intermediate_dir = self.intermediate_dir
        output_dir = self.output_dir
        imputer = self.imputer

        # LOAD PREPROCESSED TRAINING DATA SILOS AND COMBINE THEM
        round5 = pd.read_json("%s/round5_merged.json"%input_dir)
        round4 = pd.read_json("%s/round4_merged.json"%input_dir)
        round3 = pd.read_json("%s/round3_merged.json"%input_dir)
        round2 = pd.read_json("%s/round2_merged.json"%input_dir)
        round1 = pd.read_json("%s/round1_merged.json"%input_dir)
        sent_info = pd.read_json("%s/sent_info.json"%input_dir)
        round5_clean = round5[["id", "host_id", "host_response_rate", "host_acceptance_rate",
                               "host_total_listings_count", "instant_bookable",
                               "room_type", "bucket_name", "discount_asked", "nightly_price",
                               "decision", "price_agreed", "discount_agreed", "percent_agreed",
                               "calendars", "price_requested", "beds", "number_of_reviews"]]
        round4_clean = round4[["id", "host_id", "host_response_rate", "host_acceptance_rate",
                               "host_total_listings_count", "instant_bookable",
                               "room_type", "bucket_name", "discount_asked", "nightly_price",
                               "decision", "price_agreed", "discount", "percent_agreed",
                               "calendars", "price_requested", "beds", "number_of_reviews"]]
        round3_clean = round3[["id", "host_id", "host_response_rate", "host_acceptance_rate",
                               "host_total_listings_count", "instant_bookable",
                               "room_type", "bucket_name", "discount_asked", "nightly_price",
                               "decision", "price_agreed", "discount_agreed", "percent_agreed",
                               "calendars", "price_requested", "beds", "number_of_reviews"]]
        round2_clean = round2[["id", "host_id", "host_response_rate", "host_acceptance_rate",
                               "host_total_listings_count", "instant_bookable",
                               "room_type", "bucket_name", "discount_asked", "nightly_price",
                               "decision", "price_agreed", "discount_agreed", "percent_agreed",
                               "calendars", "price_requested", "beds", "number_of_reviews"]]
        round1_clean = round1[["id", "host_id", "host_response_rate", "host_acceptance_rate",
                               "host_total_listings_count", "instant_bookable",
                               "room_type", "bucket_name", "discount_asked", "nightly_price",
                               "availability", "price", "discount",
                               "calendars", "price_requested", "beds", "number_of_reviews"]]

        round1_clean = round1_clean.rename(columns = {"availability":"decision", "price":"price_agreed", "discount":"percent_agreed"})
        round4_clean = round4_clean.rename(columns = {"discount":"discount_agreed"})
        round5_clean["decision"] = [1 if val==4 else val for val in round5_clean.decision.values]

        round1_clean["percent_agreed"] = [np.nan if val==None else float(val.strip("%"))/100. for val in  round1_clean.percent_agreed.values]

        combined = pd.concat([round1_clean, round2_clean, round3_clean, round4_clean, round5_clean], keys=["round1", "round2", "round3", "round4", "round5"], ignore_index=False)

        combined["source"] = combined.index.labels[0]+1
        combined = combined.reset_index(drop=True)

        sent_info = sent_info.rename(columns = {"property_id":"id"})
        sent_info_clean = sent_info[['id', 'source', 'start', 'end', 'sent_date']]

        combined = pd.merge(combined, sent_info_clean, on=['id', 'source'], how='inner')

        # CLEAN AND CONSTRUCT X VARIABLES (ESTIMATORS)
        ### PRICE RELATED
        calendar_price =  np.array([(np.median(calendar["price_USD"].values()),
                                     np.std(calendar["price_USD"].values()),
                                     np.max(calendar["price_USD"].values()))
                                    for calendar in combined["calendars"].values])
        calendar_median_price = calendar_price[:,0]
        calendar_price_std = calendar_price[:,1]
        calendar_price_high = calendar_price[:,2]
        combined["price_median"]=calendar_median_price
        combined["price_std"]=calendar_price_std/calendar_median_price
        combined["orig_percent_off"]=1-combined["nightly_price"]/calendar_price_high
        ### OCCUPANCY
        key_1m = map(str, range(30))
        calendar_occupancy = [np.mean([calendar["availability"][key] for key in key_1m]) for calendar in combined["calendars"].values]
        combined["occupancy_1m"]=calendar_occupancy
        ### SHARED WITH HOST
        shared = combined["room_type"].values
        combined["shared"] = [x!="Entire home/apt" for x in shared]
        ### INSTANT BOOKABLE
        instant = combined["instant_bookable"].values
        combined["instant"] = [x=="t" for x in shared]
        ### RESPONSE RATE
        imputer = Imputer(copy=True, missing_values='NaN', strategy='mean', axis=1)
        response_num = np.array([float(response_rate.strip('%'))/100 for response_rate in combined["host_response_rate"].fillna(value="-100%").values])
        response_num = np.array([np.nan if x < 0 else x for x in response_num])
        response_imputed = imputer.fit_transform(response_num)[0]
        combined["response_rate"] = response_imputed
        ### CALCULATE NUMERIC BUCKET
        calc_len = lambda row: (dt.datetime.strptime(row["end"], '%Y-%m-%d') - dt.datetime.strptime(row["start"], '%Y-%m-%d')).days
        calc_adv = lambda row: np.floor((dt.datetime.strptime(row["start"], '%Y-%m-%d') - dt.datetime.strptime(row["sent_date"], '%Y-%m-%d')).days/7)
        combined["length_num"] = combined.apply(calc_len, axis=1)
        combined["advance_num"] = combined.apply(calc_adv, axis=1)
        ### BUCKETS (LONG...)
        opening_attr = combined["bucket_name"].values
        N = len(opening_attr)
        orp_1 = np.zeros(N)
        orp_2 = np.zeros(N)
        orp_3 = np.zeros(N)
        adv_1 = np.zeros(N)
        adv_2 = np.zeros(N)

        orphan = np.ones(N)

        for (i,x) in enumerate(opening_attr):
            if x == "days1_weeks1":
                orp_1[i] = 1
                adv_1[i] = 1
            elif x == "days1_weeks2":
                orp_1[i] = 1
                adv_2[i] = 1
            elif x == "days1_weeksM":
                orp_1[i] = 1
            elif x == "days2_weeks1":
                orp_2[i] = 1
                adv_1[i] = 1
            elif x == "days2_weeks2":
                orp_2[i] = 1
                adv_2[i] = 1
            elif x == "days2_weeksM":
                orp_2[i] = 1
            elif x == "days3_weeks1":
                orp_3[i] = 1
                adv_1[i] = 1
            elif x == "days3_weeks2":
                orp_3[i] = 1
                adv_2[i] = 1
            elif x == "days3_weeksM":
                orp_3[i] = 1
            elif x == "daysM_weeks1":
                adv_1[i] = 1
                orphan[i] = 0
            elif x == "daysM_weeks2":
                adv_2[i] = 1
                orphan[i] = 0
            elif x == "daysM_weeksM":
                orphan[i] = 0

        combined["orp_1"]=orp_1
        combined["orp_2"]=orp_2
        combined["orp_3"]=orp_3
        combined["adv_1"]=adv_1
        combined["adv_2"]=adv_2

        combined["orphan"] = orphan
        combined["orp_1, adv_1"] = combined["orp_1"]*combined["adv_1"]
        combined["orp_1, adv_2"] = combined["orp_1"]*combined["adv_2"]
        combined["orp_2, adv_1"] = combined["orp_2"]*combined["adv_1"]
        combined["orp_2, adv_2"] = combined["orp_2"]*combined["adv_2"]
        combined["orp_3, adv_1"] = combined["orp_3"]*combined["adv_1"]
        combined["orp_3, adv_2"] = combined["orp_3"]*combined["adv_2"]

        # CLEAN AND CONSTRUCT Y VARIABLES (TARGETS)
        ### DISCOUNT
        orig_price = combined["nightly_price"]
        price_agreed_percent = (np.nan_to_num(combined["price_agreed"])/orig_price)
        discount_agreed1 = [0 if (d>=1 or d==0) else 1-d for d in price_agreed_percent]
        discount_agreed2 = np.nan_to_num(combined["discount_agreed"]/orig_price)
        discount_agreed3 = np.nan_to_num(combined["percent_agreed"])
        discount_obtained = np.max([discount_agreed1, discount_agreed2, discount_agreed3], axis=0)
        combined["discount_obtained"] = discount_obtained

        # CREATE TRAINING SAMPLES
        mask_available = combined["decision"] != -1
        mask_agreed = combined["discount_obtained"] != 0
        mask_no_source1 = combined["source"] != 1
        combined_available = combined[mask_available]
        combined_agreed = combined[mask_agreed * mask_no_source1]
        cal_param_1 = ["orp_1", "orp_2", "orp_3", "adv_1", "adv_2"]
        cal_param_1_num = ["orphan", "length_num", "advance_num"]
        cal_param_2 = ["orp_1, adv_1", "orp_1, adv_2", "orp_2, adv_1", "orp_2, adv_2", "orp_3, adv_1", "orp_3, adv_2"]
        host_param = ["host_total_listings_count", "response_rate", "instant", "shared", "beds", "number_of_reviews",
                      "price_median", "price_std", "occupancy_1m", "orig_percent_off"]
        param_rf = cal_param_1_num + host_param
        param_lr = cal_param_1_num + host_param + cal_param_2

        X_responded_rf = combined_available[param_rf].values
        X_responded_lr = combined_available[param_lr].values
        Y_responded = combined_available["decision"].values

        X_all_rf = combined[param_rf].values
        X_all_lr = combined[param_lr].values
        Y_all = combined["decision"].values
        Y_all = np.max(zip(Y_all,np.zeros(len(Y_all))), axis=1)

        X_discount_rf = combined_agreed[param_rf].values
        X_discount_lr = combined_agreed[param_lr].values
        Y_discount = combined_agreed["discount_obtained"].values

        # CREATE PLOT LABELS
        features = ["orp_1", "orp_2", "orp_3", "adv_1", "adv_2",
                    "orphan", "length_num", "advance_num",
                    "orp_1, adv_1", "orp_1, adv_2", "orp_2, adv_1", "orp_2, adv_2", "orp_3, adv_1", "orp_3, adv_2",
                    "host_total_listings_count", "response_rate", "instant", "shared", "beds", "number_of_reviews",
                    "price_median", "price_std", "occupancy_1m", "orig_percent_off"]

        labels = ["1-day orphan", "2-day orphan", "3-day_orphan", "Within 1 week", "1-2 weeks in advance",
                  "Orphan day", "Length of stay", "Weeks in advance",
                  "1-day orphan within 1 week", "1-day orphan 1-2 weeks in advance",
                  "2-day orphan within 1 week", "2-day orphan 1-2 weeks in advance",
                  "3-day orphan within 1 week", "3-day orphan 1-2 weeks in advance",
                  "Host listing count", "Host response rate", "Instant bookable", "Space shared with host", "Number of beds", "Number of reviews",
                  "Usual price", "Price variation", "1-month occupancy", "Off Highest Price"]

        label_dict = dict(zip(features, labels))

        # CLASSIFY NEGOTIABLE HOSTS WITH RANDOM FOREST
        clf_all_rf = RandomForestClassifier(n_estimators=200,
                                        max_features=7,
                                        max_depth=None,
                                        min_samples_split=1,
                                        n_jobs=-1,
                                        warm_start=True,
                                        class_weight = "balanced")
        clf_all_rf.fit(X_all_rf, Y_all)

        # CLASSIFY ANTICIPATED DISCOUNT WITH RANDOM FOREST
        clf_discount_rf = RandomForestRegressor(n_estimators=50,
                                        max_features=7,
                                        max_depth=None,
                                        min_samples_split=1,
                                        n_jobs=-1,
                                        warm_start=True)
        clf_discount_rf.fit(X_discount_rf, Y_discount)

        # SAVING THE MODEL TO EXTERNAL FILE
        with open('%s/clf_all_rf.pkl'%intermediate_dir, 'wb') as f:
            pickle.dump(clf_all_rf, f)
        with open('%s/clf_discount_rf.pkl'%intermediate_dir, 'wb') as f:
            pickle.dump(clf_discount_rf, f)

        return clf_all_rf, clf_discount_rf
    #################################################################################################################


    #################################################################################################################
    def Preprocess(self):
        # INITIALIZE VARIABLES
        city = self.city
        today_parsed = self.today_parsed
        input_dir = self.input_dir
        intermediate_dir = self.intermediate_dir
        output_dir = self.output_dir
        calendar_date = self.calendar_date
        calendar_date_parsed = self.calendar_date_parsed
        imputer = self.imputer

        inside_raw = pd.read_csv("%s/listings_%s.csv"%(input_dir, city))
        calendar_raw = pd.read_json("%s/calendar_%s_%s.json"%(input_dir, city, calendar_date))

        # CALENDAR: PARSE CALENDAR DATA INTO DATAFRAME
        parsed_calendars = {}
        for i in calendar_raw.keys():
            parsed_calendars[i] = self.parse_calendar(calendar_raw[i], calendar_date_parsed)
        listing_id = [int(key) for key in parsed_calendars.keys()]
        df_calendar = pd.DataFrame({'id': listing_id, 'calendars': parsed_calendars.values()})

        # CALENDAR: ADD COLUMNS FOR FUTURE CALCULATION
        ### AVAILABILITY
        calendar_available = [calendar["availability"].values for calendar in df_calendar['calendars']]
        df_calendar["availabilities"] = calendar_available
        ### MIN NIGHTS
        calendar_min_nights = [np.max(calendar["min_nights"].values) for calendar in df_calendar['calendars']]
        df_calendar["min_nights"] = calendar_min_nights


        ### PRICE RELATED
        calendar_price = [calendar["price_USD"].values for calendar in df_calendar['calendars']]
        df_calendar["prices"] = calendar_price
        df_calendar["price_median"] = np.median(calendar_price, axis=1)
        df_calendar["price_high"] = np.max(calendar_price, axis=1)
        df_calendar["price_std"] = np.std(calendar_price, axis=1)/df_calendar["price_median"]


        ### 1M OCCUPANCY
        key_1m = range(30)
        calendar_occupancy = [np.mean([calendar["availability"][key] for key in key_1m])
                              for calendar in df_calendar["calendars"].values]
        df_calendar["occupancy_1m"]=calendar_occupancy


        # INSIDEAIRBNB: PREPROCESS COLUMNS
        ### SHARED
        shared = inside_raw["room_type"].values
        inside_raw["shared"] = [x!="Entire home/apt" for x in shared]
        ### INSTANT
        instant = inside_raw["instant_bookable"].values
        inside_raw["instant"] = [x=="t" for x in shared]
        ### RESPONSE RATE
        response_num = np.array([float(response_rate.strip('%'))/100
                                 for response_rate in inside_raw["host_response_rate"].fillna(value="-100%").values])
        response_num = np.array([np.nan if x < 0 else x for x in response_num])
        response_imputed = imputer.fit_transform(response_num)[0]
        inside_raw["response_rate"] = response_imputed

        # SELECT USEFUL COLUMNS FROM INSIDEAIRBNB DATA
        inside_col = [u'id', u'response_rate', u'host_is_superhost', u'host_total_listings_count',
                      u'number_of_reviews', u'instant', u'shared', u'beds']
        df_listing = inside_raw[inside_col]
        imputer = self.imputer


        # MERGE CALENDAR WITH INSIDEAIRBNB DATA
        df_merged = pd.merge(df_calendar, df_listing, on='id', how='inner')

        self.df_merged = df_merged

        df_merged.to_json("%s/preprocessed_calendar_%s.json" %(intermediate_dir, calendar_date))
        return df_merged

    # UTILITY FUNCTIONS FOR PREPROCESS
    def parse_calendar(self, calendar, date_scraped):
        date = []
        price_USD = []
        availability = []
        min_nights = []
        day_list = []
        for month in calendar['calendar_months']:
            for day in month['days']:
                day_parsed = dt.datetime.strptime(day['date'], '%Y-%m-%d')
                if (day_parsed > date_scraped) & (day_parsed not in day_list):
                    date.append(day['date'])
                    price_USD.append(day['price']['native_price'])
                    availability.append(day['available'])
                    min_nights.append(month['condition_ranges'][0]['conditions'][u'min_nights'])
                day_list.append(day_parsed)
        return pd.DataFrame({'date':date, 'price_USD': price_USD, 'availability':availability, 'min_nights': min_nights})
    #################################################################################################################



    #################################################################################################################
    def Predict(self, check_in, check_out):
        input_dir = self.input_dir
        intermediate_dir = self.intermediate_dir
        output_dir = self.output_dir
        today_parsed = self.today_parsed
        calendar_date = self.calendar_date
        calendar_date_parsed = self.calendar_date_parsed
        imputer = self.imputer

        df_merged = pd.read_json("%s/preprocessed_calendar_%s.json" %(intermediate_dir, calendar_date))

        with open('%s/clf_all_rf.pkl'%intermediate_dir, 'rb') as f:
            clf_all = pickle.load(f)
        with open('%s/clf_discount_rf.pkl'%intermediate_dir, 'rb') as f:
            clf_discount = pickle.load(f)

        # PREPROCESS CHECKIN AND CHECKOUT DATE
        check_in_parsed = dt.datetime.strptime(check_in, '%Y-%m-%d')
        check_out_parsed = dt.datetime.strptime(check_out, '%Y-%m-%d')

        # NARROW BOOKABLE LISTINGS
        advance = (check_in_parsed - today_parsed).days + 1
        length = (check_out_parsed - check_in_parsed).days
        mask_bookable = df_merged.apply(lambda row: 1
                                        if (np.sum(row['availabilities'][advance-1: advance-1+length]) == length
                                            & row['min_nights'] <= length)
                                        else 0, axis=1)
        df_bookable = df_merged[mask_bookable==1]
        orphan = df_bookable.apply(lambda row: row['availabilities'][advance-2]==False & row['availabilities'][advance-1+length]==False, axis=1)
        count_bookable = len(df_bookable)

        # CALCULATE ALREADY DISCOUNTED
        df_bookable["nightly_price"] = df_bookable.apply(lambda row: np.mean(row['prices'][advance-1: advance-1+length]), axis=1)
        df_bookable["orig_percent_off"] = 1 - df_bookable["nightly_price"]/df_bookable["price_high"]

        # CALCULATE TRIP DETAILS
        # CREATE TEST SET
        X_trip = np.zeros((count_bookable,3))
        X_trip[:,0] = orphan
        X_trip[:,1] = length
        X_trip[:,2] = advance
        X_listing = imputer.fit_transform(df_bookable[["host_total_listings_count", "response_rate", "instant", "shared", "beds", "number_of_reviews",
                                                                        "price_median", "price_std", "occupancy_1m", "orig_percent_off"]].values)
        X_test = np.hstack((X_trip, X_listing))

        predict_all = clf_all.predict_proba(X_test)[:,1]
        predict_discount = clf_discount.predict(X_test)

        return zip(df_bookable["id"].values, df_bookable["orig_percent_off"].values, predict_all, predict_discount)

    #################################################################################################################