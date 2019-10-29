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
import multiprocessing
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.decomposition import PCA



#import custom defined functions
os.chdir('H:/Personal/Columbia/STAT 5243 Applied Data Science/Fall2019-proj3-sec1--grp5/doc')
#os.chdir('D:/wwyws/Documents/Columbia/STAT5243 Applied Data Science/Fall2019-proj3-sec1--grp5/doc')
sys.path.append("..")
import lib.feature as ft


info = pd.read_csv('../data/train_set/label.csv',usecols = range(1,6))

#%%parameters
rand_seed = 123
test_size = 0.2
cv_folds = 5
num_cores = multiprocessing.cpu_count()

#%%train/test split
n = info.shape[0]
n_train = round(n*4/5)
train_idx = np.random.choice(info['Index'], size=n_train, replace=False)
test_idx = np.setdiff1d(info['Index'].values,train_idx,assume_unique=True)
train_idx += -1
test_idx += -1
all_idx = np.array(info.index)

##%%loading and processing fiducial points
##def readMat(index):
##    thisMat = loadmat('../data/train_set/points/' + '%04d' % index + '.mat')
##    return pd.DataFrame(round(pd.DataFrame(thisMat[list(thisMat)[3]]),0))
##
##fiducial_pt_list = list(map(readMat, list(range(1,2501))))
##f = open('../output/fiducial_pt_list', 'wb')
##pickle.dump(fiducial_pt_list, f)
##f.close()
#
f = open('../output/fiducial_pt_list', 'rb')
fiducial_pt_list = pickle.load(f)
f.close()


#
#index = ['{0:04}'.format(num) for num in range(1, 2501)]
#
#mats = []
#for ind in index:
#    temp = loadmat( '../data/train_set/points/' + ind + ".mat")
#    mats.append(temp[[*temp][-1]])
#    
#mats = [mat.round() for mat in mats]
#
#feature_slope_df = pd.DataFrame(ft.feature_slope(fiducial_pt_list))
#feature_slope_df.columns = ["feature"+ str(num) for num in range(1, 3004)]
#a = ft.feature_dist_slope(copy.deepcopy(mats),info)
#%%extract features
#dat_train = ft.feature(copy.deepcopy(fiducial_pt_list),train_idx,info)
#dat_test = ft.feature(copy.deepcopy(fiducial_pt_list),test_idx,info)
dat_full = ft.feature(copy.deepcopy(fiducial_pt_list),all_idx,info)
dat_full_new = ft.feature_dist_slope(copy.deepcopy(fiducial_pt_list),info)
#%%X Y Split
X, y = dat_full.iloc[:,:-1],dat_full.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_seed)

X_new, y_new = dat_full_new.iloc[:,:-1],dat_full_new.iloc[:,-1]
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=test_size, random_state=rand_seed)

pca = PCA(n_components = 0.99)
X_pca = pca.fit_transform(X_new)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_new, test_size=test_size, random_state=rand_seed)


#%% baseline GBM model. 
baseGBM = GradientBoostingClassifier(random_state=rand_seed,max_depth = 1)
startTime = time.time()
baseGBM.fit(X_train, y_train)
endTime= time.time()-startTime
baseGBM_prediction = baseGBM.predict(X_test)
baseGBM_accuracy = accuracy_score(y_test,baseGBM_prediction)

baseGBM = GradientBoostingClassifier(random_state=rand_seed,max_depth = 1)
baseGBMFitStart =  time.time()
baseGBM.fit(X_train_new, y_train_new)
print("Base GBM Fit Time Is: " + str(time.time()-baseGBMFitStart))  #Fit time around 165 seconds
baseGBM_prediction = baseGBM.predict(X_test_new)
baseGBM_accuracy = accuracy_score(y_test_new,baseGBM_prediction)



baseGBM = GradientBoostingClassifier(random_state=rand_seed,max_depth = 1)
baseGBMFitStart =  time.time()
baseGBM.fit(X_train_pca, y_train_pca)
print("Base GBM Fit Time Is: " + str(time.time()-baseGBMFitStart))  #Fit time around 165 seconds
baseGBM_prediction = baseGBM.predict(X_test_pca)
baseGBM_accuracy = accuracy_score(y_test_new,baseGBM_prediction)


#save the model
baseGBMOut = {'Model':baseGBM, 'accuracy':baseGBM_accuracy}
f = open('../output/baseGBMOut', 'wb')
pickle.dump(baseGBMOut, f)
f.close()

f = open('../output/baseGBMOut', 'rb')
baseGBMOut = pickle.load(f)
f.close()


#%%xgboost
#xgbModel = xgboost.XGBClassifier(random_state = rand_seed)
#xgbFitStart = time.time()
#xgbModel.fit(X_train, y_train)
#print("XGBOOST Fit Time Is: " + str(time.time()-xgbFitStart))
#xgb_prediction = xgbModel.predict(X_test)
#xgb_accuracy  = accuracy_score(y_test,xgb_prediction)
##save the model
#xgbOut = {'Model':xgbModel, 'accuracy':xgb_accuracy}
#f = open('../output/xgbOut', 'wb')
#pickle.dump(xgbOut, f)
#f.close()

f = open('../output/xgbOut', 'rb')
xgbOut = pickle.load(f)
f.close()
xgbModel = xgbOut['Model'] ##original accuracy 0.486


#start tuning parameters, first is n_estimators i.e. the number of boosting operations
xgb1 = xgboost.XGBClassifier(
    colsample_bytree = 1, ##different from 0.8 for now
    gamma = 0,
    learning_rate = 0.1,
    max_depth =3, ##different from 5 for now
    min_child_weight = 1,
    n_estimators = 1000,
    objective = 'multi:softmax',
    scale_pos_weight = 1,
    random_state = rand_seed,
    subsample = 1,  ##different from 0.8 for now
    num_class = 22,
    nthread = 6
)


xgb_param = xgb1.get_xgb_params()
xgtrain = xgboost.DMatrix(X_train, label=y_train-1)

startTime = time.time() 
cvresult = xgboost.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=cv_folds,
            metrics = 'mlogloss',early_stopping_rounds=50)
print("CV Time Is: " + str(time.time()-startTime))
xgb1.set_params(n_estimators=cvresult.shape[0]) #115

xgb1.fit(X_train,y_train)
xgb_prediction = xgb1.predict(X_test)
xgb_accuracy  = accuracy_score(y_test,xgb_prediction) #increased to 0.49

##next we tune max_depth and min_child_weight
param_test1 = {
 'max_depth':range(3,10,1),
 'min_child_weight':range(1,6,1)
}
xgb2 = xgboost.XGBClassifier(
    colsample_bytree = 1, ##different from 0.8 for now
    gamma = 0,
    learning_rate = 0.1,
    max_depth =3, ##different from 5 for now
    min_child_weight = 1,
    n_estimators = 115, #now we've set this to 115, the optimized parameter from the last tuning
    objective = 'multi:softmax',
    scale_pos_weight = 1,
    random_state = rand_seed,
    subsample = 1,  ##different from 0.8 for now
    num_class = 22,
    nthread = 6
)


gsearch1 = GridSearchCV(estimator = xgb2, 
 param_grid = param_test1, scoring='accuracy',n_jobs=6,iid=False, cv=5)
startTime = time.time()
gsearch1.fit(X_train,y_train)
print("grid Fit Time Is: " + str(time.time()-startTime))
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


xgb1 = xgboost.XGBClassifier(
    colsample_bytree = 1, ##different from 0.8 for now
    gamma = 0,
    learning_rate = 0.1,
    max_depth =3, ##different from 5 for now
    min_child_weight = 1,
    n_estimators = 115,
    objective = 'multi:softmax',
    scale_pos_weight = 1,
    random_state = rand_seed,
    subsample = 1,  ##different from 0.8 for now
    num_class = 22,
    nthread = num_cores
)


startTime = time.time()
xgb1.fit(X_train,y_train)
print("XGBOOST Fit Time Is: " + str(time.time()-startTime))
xgb_prediction = xgb1.predict(X_test)
xgb_accuracy  = accuracy_score(y_test,xgb_prediction) #increased to 0.49



#%%lightgbm
#lgb_train = lgb.Dataset(X_train, label = y_train-1)
#lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
#lgb_params = {
#    'boosting_type': 'gbdt',
#    'objective': 'multiclass',
#    'num_class': 22,
#    'learning_rate' : 0.1,
#    'seed' : rand_seed
#}
#lgbFitStart = time.time()
#lgbModel = lgb.train(lgb_params,train_set = lgb_train)
#print("LightGBM Fit Time Is: " + str(time.time()-lgbFitStart))
#lgb_prediction = np.argmax(lgbModel.predict(X_test),axis = 1)+1
#lgb_accuracy  = accuracy_score(y_test,lgb_prediction)
###save the model
#lgbOut = {'Model':lgbModel, 'accuracy':lgb_accuracy}
#f = open('../output/lgbOut', 'wb')
#pickle.dump(lgbOut, f)
#f.close()

f = open('../output/lgbOut', 'rb')
lgbOut = pickle.load(f)
f.close()

lgb1 = lgb.LGBMClassifier(
    boosting_type = 'gbdt',
    objective = 'multiclass',
    num_class = 22,
    learning_rate = 0.1,
    random_state = rand_seed,
    n_jobs = num_cores
        )
startTime = time.time()
lgb1.fit(X_train,y_train)
print("LGB Fit Time Is: " + str(time.time()-startTime))
lgb_prediction = lgb1.predict(X_test)
lgb_accuracy  = accuracy_score(y_test,lgb_prediction)



lgb1 = lgb.LGBMClassifier(
    boosting_type = 'gbdt',
    objective = 'multiclass',
    num_class = 22,
    learning_rate = 0.1,
    random_state = rand_seed,
    n_jobs = num_cores
        )
startTime = time.time()
lgb1.fit(X_train_new,y_train_new)
print("LGB Fit Time Is: " + str(time.time()-startTime))
lgb_prediction = lgb1.predict(X_test_new)
lgb_accuracy  = accuracy_score(y_test_new,lgb_prediction)














