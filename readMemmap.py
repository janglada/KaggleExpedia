__author__ = 'joan'
import numpy as np
from numpy.lib.format import open_memmap
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from joblib import Parallel, delayed

import csv
import time


train = open_memmap('data/train.npy', mode='r',  dtype=np.float32, shape=(37670294, 6))
test  = open_memmap('data/test.npy',  mode='r',  dtype=np.float32, shape=(2528243, 7))

def get_max_prob( x ):
    return x.argsort()[-5:][::-1]

# def predict(row, output, i):
#     X = row[1:7]
#     id =  int(row[0])
#     Y = int(Forest.predict(X.reshape(1, -1)))
#     output[i, 0] = id
#     output[i, 0] = Y
#     if i%1000 == 0:
#         print 'processing row', i

#train_cols  = ['srch_destination_id','site_name','user_location_city','is_package', 'hotel_country' 'sum', 'count', 'hotel_cluster']
#test_cols   = [ 'id', 'srch_destination_id','site_name','user_location_city','is_package', 'hotel_country']


# print np.shape(train)
# print train[1,:]
# print train[1,1:8]
# print train[1,7]
#

mask = np.random.choice([False, True], len(train), p=[0.99, 0.01])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(
   train[mask, 1:6], train[mask, 5], test_size=0.6, random_state=0)

print 'Data tes split done'
Forest = RandomForestClassifier(n_jobs=4, n_estimators=20)

print 'fitting....'
Forest = Forest.fit(X_train, y_train)
print 'fitting....done'
# score = Forest.score(X_test, y_test)
# print 'score done....done'
# print('-' * 38)
# print 'score'
# print score
# print('-' * 38)

#joblib.dump(Forest, 'RandomForestClassifier.pkl')

#https://pythonhosted.org/joblib/parallel.html
print('Writting submission file')
rows = np.shape(test)[0]
cols = np.shape(test)[1]
print  np.shape(test)
fieldnames = ['id','hotel_cluster']


# Pre-allocate a writeable shared memory map as a container for the
# results of the parallel computation
# out = np.memmap("out.npy", dtype=np.int, shape=(rows,2), mode='w+')

# Fork the worker processes to perform computation concurrently
# Parallel(n_jobs=4)(delayed(predict)(test[i,:], out, i) for i in range(rows))

# step = 70229
step = 2
rows = 10
with open('submission.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(fieldnames)
    for i in np.arange(0, rows, step):
        # if (i% 1000 == 0):

        #print i
        X   = np.array(test[i:i+step,  1:7])
        id  = test[i:i+step, 0]
        p =  Forest.predict_proba(X)
        Y = Forest.predict(X)
        # clusters  = p[0,:].argsort()[-5:][::-1]
        z = np.apply_along_axis( get_max_prob, axis=1, arr=p )
        print z
        print  np.shape(id), np.shape(z), np.shape(z.T)
        # start = time.clock();

        # print 't predict data = ', (time.clock() -start)
        res =  np.column_stack((id.astype(int).T,z.astype(int).T))
        print res
        spamwriter.writerows(res)
        # print 't write data = ', (time.clock() -start)


        #print test[i,:]
print('Writting submission file.. DONE')
print 'Memmap files read'