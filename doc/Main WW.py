import os
import copy
import pandas as pd
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost
import time
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm

#import custom defined functions
#os.chdir('H:/Personal/Columbia/STAT 5243 Applied Data Science/Fall2019-proj3-sec1--grp5/doc')
os.chdir('D:/wwyws/Documents/Columbia/STAT5243 Applied Data Science/Fall2019-proj3-sec1--grp5/doc')
sys.path.append("..")
import lib.feature as ft





info = pd.read_csv('../data/train_set/label.csv',usecols = range(1,6))

#%%parameters
rand_seed = 123
test_size = 0.2

#%%train/test split
n = info.shape[0]
n_train = round(n*4/5)
train_idx = np.random.choice(info['Index'], size=n_train, replace=False)
test_idx = np.setdiff1d(info['Index'].values,train_idx,assume_unique=True)
train_idx += -1
test_idx += -1
all_idx = np.array(info.index)

#%%loading and processing fiducial points
#def readMat(index):
#    thisMat = loadmat('../data/train_set/points/' + '%04d' % index + '.mat')
#    return pd.DataFrame(round(pd.DataFrame(thisMat[list(thisMat)[3]]),0))
#
#fiducial_pt_list = list(map(readMat, list(range(1,2501))))
#f = open('../output/fiducial_pt_list', 'wb')
#pickle.dump(fiducial_pt_list, f)
#f.close()

f = open('../output/fiducial_pt_list', 'rb')
fiducial_pt_list = pickle.load(f)
f.close()

#%%extract features
#dat_train = ft.feature(copy.deepcopy(fiducial_pt_list),train_idx,info)
#dat_test = ft.feature(copy.deepcopy(fiducial_pt_list),test_idx,info)
dat_full = ft.feature(copy.deepcopy(fiducial_pt_list),all_idx,info)
#%%X Y Split
X, y = dat_full.iloc[:,:-1],dat_full.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_seed)

#%% baseline GBM model
#baseGBM = GradientBoostingClassifier(random_state=rand_seed)
#baseGBMFitStart =  time.time()
#baseGBM.fit(X_train, y_train)
#print("Base GBM Fit Time Is: " + str(time.time()-baseGBMFitStart))
##save the model
#f = open('../output/baseGBM', 'wb')
#pickle.dump(baseGBM, f)
#f.close()
f = open('../output/baseGBM', 'rb')
baseGBM = pickle.load(f)
f.close()
baseGBM_prediction = baseGBM.predict(X_test)
baseGBM_accuracy = accuracy_score(y_test,baseGBM_prediction)


#%%xgboost
xgbModel = xgboost.XGBClassifier(random_state = rand_seed)
xgbFitStart = time.time()
xgbModel.fit(X_train, y_train)
print("XGBOOST Fit Time Is: " + str(time.time()-xgbFitStart))
#save the model
f = open('../output/xgbModel', 'wb')
pickle.dump(xgbModel, f)
f.close()
f = open('../output/xgbModel', 'rb')
xgbModel = pickle.load(f)
f.close()
xgb_prediction = xgbModel.predict(X_test)
xgb_accuracy  = accuracy_score(y_test,xgb_prediction)



#%%lightgbm
lgb_train = lgb.Dataset(X_train, label = y_train-1)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 22,
    'learning_rate' : 0.1,
    'seed' : rand_seed
}
lgbFitStart = time.time()
lgbModel = lgb.train(lgb_params,train_set = lgb_train)
print("LightGBM Fit Time Is: " + str(time.time()-lgbFitStart))

lgb_prediction = np.argmax(lgbModel.predict(X_test),axis = 1)+1
lgb_accuracy  = accuracy_score(y_test,lgb_prediction)

















