import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from random import sample
import numpy as np


store_train = pd.read_hdf('data/expedia_agg_many.h5',     'expedia_agg_many');
store_test  = pd.read_hdf('data/expedia_test_agg_many.h5','expedia_test_agg_many');

print 'Datasets loaded'


store_train_sample              = store_train.loc[sample(store_train.index, 100)]
store_test_sample               = store_train.loc[sample(store_train.index, 100)]

store_test_submission_sample    = store_test.loc[sample(store_train.index, 100)]





features = store_train_sample.columns[:5]
Forest = RandomForestClassifier(n_jobs=2)
y = store_train_sample['hotel_cluster']
print 'training'
Forest = Forest.fit(store_train_sample[features], y)

preds = Forest.predict(store_test_sample[features])

np.savetxt('data/submission2.csv',preds , delimiter=',', fmt='%f')

print(pd.crosstab(store_test_sample['hotel_cluster'], preds,
                  rownames=['actual'],
                  colnames=['preds']))

