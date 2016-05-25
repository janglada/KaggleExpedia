import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score


def cleanChunk(df):
    #df['srch_ci_month'] = pd.DatetimeIndex(df['srch_ci']).month
    #df['srch_ci_year']  = pd.DatetimeIndex(df['srch_ci']).year
    #df['num_days'] = (df['srch_co'] - df['srch_ci']) / np.timedelta64(1, 'D')

    df = df.drop('srch_ci', 1)
    df = df.drop('srch_co', 1)
    df = df.drop('date_time', 1)
    df = df.drop('user_id', 1)
    return df


def readCSV(filename):


    return train

def readTrainData(cols):

    # date_time,site_name,posa_continent,user_location_country,user_location_region,user_location_city,orig_destination_distance,user_id,is_mobile,is_package,channel,srch_ci,srch_co,srch_adults_cnt,srch_children_cnt,srch_rm_cnt,srch_destination_id,srch_destination_type_id,is_booking,cnt,hotel_continent,hotel_country,hotel_market,hotel_cluster
    train = pd.read_csv(filename, header=True,
                        names=['date_time', 'site_name', 'posa_continent', 'user_location_user',
                               'country_location_region', 'user_location_city', 'orig_destination_distance', 'user_id',
                               'is_mobile',
                               'is_package', 'channel', 'srch_ci', 'srch_co', 'srch_adults_cnt', 'srch_children_cnt',
                               'srch_rm_cnt',
                               'srch_destination_id', 'srch_destination_type_id', 'is_booking', 'cnt', 'hotel_continent',
                               'hotel_country', 'hotel_market', 'hotel_cluster']
                        , parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=50000)
    aggs = []
    print('-' * 38)
    for chunk in train:
        #print chunk
        chunk = cleanChunk(chunk);
        agg = chunk.groupby(['srch_destination_id','site_name','user_location_city','is_package', 'hotel_country', 'hotel_cluster'])['is_booking'].agg(['sum', 'count'])
        agg.reset_index(inplace=True)
        aggs.append(agg)
        #print('.',end='')
    print('')
    aggs = pd.concat(aggs, axis=0)
    aggs.head()
    print aggs.head();

    aggs.to_hdf('data/expedia_agg_many.h5','expedia_agg_many')


def readTestData():
    # date_time,site_name,posa_continent,user_location_country,user_location_region,user_location_city,orig_destination_distance,user_id,is_mobile,is_package,channel,srch_ci,srch_co,srch_adults_cnt,srch_children_cnt,srch_rm_cnt,srch_destination_id,srch_destination_type_id,is_booking,cnt,hotel_continent,hotel_country,hotel_market,hotel_cluster
    train = pd.read_csv("data/test.csv", header=True,
                        names=['id','date_time','site_name','posa_continent','user_location_country','user_location_region','user_location_city','orig_destination_distance','user_id','is_mobile','is_package','channel','srch_ci','srch_co','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_id','srch_destination_type_id','hotel_continent','hotel_country','hotel_market']
                        , parse_dates=['date_time', 'srch_ci', 'srch_co'], chunksize=50000)

    cols = ['id', 'srch_destination_id','site_name','user_location_city','is_package', 'hotel_country']

    aggs = []
    print('-' * 38)
    for chunk in train:
        agg = chunk[cols]
        agg.reset_index(inplace=True)
        aggs.append(agg)
        #print('.',end='')
    print('')
    aggs = pd.concat(aggs, axis=0)
    aggs.head()
    print aggs.head();

    aggs.to_hdf('data/expedia_test_agg_many.h5','expedia_test_agg_many')




#useful columns

readTestData()