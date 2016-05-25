__author__ = 'joan'
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

import csv


def get_max_prob( x ):
    return x.argsort()[-5:][::-1]

# =============================================================================
# TRAIN
# =============================================================================
def trainRandomForest():
    # date_time,site_name,posa_continent,user_location_country,user_location_region,user_location_city,orig_destination_distance,user_id,is_mobile,is_package,channel,srch_ci,srch_co,srch_adults_cnt,srch_children_cnt,srch_rm_cnt,srch_destination_id,srch_destination_type_id,is_booking,cnt,hotel_continent,hotel_country,hotel_market,hotel_cluster
    train = pd.read_csv("data/train.csv", header=0,
                        names=['date_time', 'site_name', 'posa_continent', 'user_location_user',
                               'country_location_region', 'user_location_city', 'orig_destination_distance', 'user_id',
                               'is_mobile',
                               'is_package', 'channel', 'srch_ci', 'srch_co', 'srch_adults_cnt', 'srch_children_cnt',
                               'srch_rm_cnt',
                               'srch_destination_id', 'srch_destination_type_id', 'is_booking', 'cnt',
                               'hotel_continent',
                               'hotel_country', 'hotel_market', 'hotel_cluster']
                        , parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=50000)


    clf = linear_model.SGDClassifier(loss='log', penalty="elasticnet", n_iter=70,n_jobs=3)

    n = 0;
    rfs = [];
    print('-' * 38)
    cls =  np.arange(100)
    # http://stackoverflow.com/questions/28489667/combining-random-forest-models-in-scikit-learn
    for chunk in train:
        agg = chunk.groupby(['srch_destination_id', 'site_name', 'user_location_city', 'is_package', 'hotel_country', 'hotel_cluster'])['is_booking'].agg(['sum', 'count'])
        agg.reset_index(inplace=True)
        X_train = agg[['srch_destination_id', 'site_name', 'user_location_city', 'is_package', 'hotel_country']]
        y_train = agg['hotel_cluster']
        # agg.reset_index(inplace=True)
        # Forest = RandomForestClassifier(n_jobs=4, n_estimators=20, n_classes_)
        # Forest = Forest.fit(X_train, y_train)
        # rfs.append(Forest)

        clf.partial_fit(X_train, y_train, classes= cls)
        print n
        n = n + 1
        # break
    print('')

    return clf


def myfunc(a):
    x = np.char.mod('%i', a)
    return " ".join(x)
# =============================================================================
# TEST
# =============================================================================
def testPrediction(clf):

        # date_time,site_name,posa_continent,user_location_country,user_location_region,user_location_city,orig_destination_distance,user_id,is_mobile,is_package,channel,srch_ci,srch_co,srch_adults_cnt,srch_children_cnt,srch_rm_cnt,srch_destination_id,srch_destination_type_id,is_booking,cnt,hotel_continent,hotel_country,hotel_market,hotel_cluster
    test = pd.read_csv("data/test.csv", header=0,  names=['id','date_time','site_name','posa_continent','user_location_country','user_location_region','user_location_city','orig_destination_distance','user_id','is_mobile','is_package','channel','srch_ci','srch_co','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_id','srch_destination_type_id','hotel_continent','hotel_country','hotel_market']
                        , parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=1000)
    with open('submission.csv', 'wb') as csvfile:
        fieldnames = ['id','hotel_cluster']
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(fieldnames)
        for chunk in test:
            cols = ['id', 'srch_destination_id','site_name','user_location_city','is_package', 'hotel_country']
            # chunk.reset_index(inplace=True)
            X = chunk[['srch_destination_id','site_name','user_location_city','is_package', 'hotel_country']]
            id  = chunk[['id']]
            Y = clf.predict(X)
            p =  clf.predict_proba(X)


            z = np.apply_along_axis( get_max_prob, axis=1, arr=p )
            z =  np.apply_along_axis(myfunc, 1, z)
            res =  np.column_stack((id,z))

            # print res
            spamwriter.writerows(res)
            # break


if __name__ == "__main__":
    clf = trainRandomForest()
    testPrediction(clf)

