#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 09:50:53 2018

@author: mohsin
"""


# Based on Bojan's -> https://www.kaggle.com/tunguz/more-effective-ridge-lgbm-script-lb-0-44944
# Changes:
# 1. Split category_name into sub-categories
# 2. Parallelize LGBM to 4 cores
# 3. Increase the number of rounds in 1st LGBM
# 4. Another LGBM with different seed for model and training split, slightly different hyper-parametes.
# 5. weights on ensemble
# 6. Add SGDRegressor

import gc
import time
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn import metrics

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.preprocessing import LabelBinarizer, MaxAbsScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, make_union

import lightgbm as lgb

from sklearn.base import BaseEstimator, TransformerMixin
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, thresh=0, func=np.mean, add_to_orig=False):
        self.cols = cols
        self.thresh = thresh
        self.func = func
        self.add_to_orig = add_to_orig
    
    #@numba.jit        
    def fit(self, X, y):
        self.prior = self.func(y)
        self._dict = {}
        for col in self.cols:
            if isinstance(col, (list, tuple)):
                print('here')
                tmp_df = X.loc[: ,col]
                col = tuple(col)
            else:
                tmp_df = X.loc[: ,[col]]
            tmp_df['y'] = y
            print(tmp_df.columns)
            #tmp_df = pd.DataFrame({'eval_col':X[col].values, 'y':y})
            if isinstance(col, (list, tuple)):
                print('here')
                col = tuple(col)
            self._dict[col] = tmp_df.groupby(col)['y'].apply(lambda x: 
                                self.func(x) if len(x) >= self.thresh  else self.prior).to_dict()
                                
            del tmp_df
        return self
    #@numba.jit
    def transform(self, X, y=None):
        X_transformed = []
        for col in self.cols:
            
            if isinstance(col, (list, tuple)):
                tmp_df = X.loc[:, col]
                enc = tmp_df[col].apply(lambda x: self._dict[tuple(col)][tuple(x)]
                                                                     if tuple(x) in self._dict[tuple(col)]
                                                                     else self.prior, axis=1).values
            else:
                tmp_df = X.loc[:, [col]]
                enc = tmp_df[col].apply(lambda x: self._dict[col][x]
                                                                     if x in self._dict[col]
                                                                     else self.prior).values
            del tmp_df
            X_transformed.append(enc)
        
        X_transformed = np.vstack(X_transformed).T
        
        if self.add_to_orig:
            return np.concatenate((X.values, X_transformed), axis=1)
            
        else:
            return X_transformed
            


NUM_BRANDS = 6000
NUM_CATEGORIES = 1000
NAME_MIN_DF = 10
MAX_FEATURES_ITEM_DESCRIPTION = 20000

def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

rmse_sklearn = metrics.make_scorer(rmse, greater_is_better=False)   
def rmsle(y, y0):
     assert len(y) == len(y0)
     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
    
def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
    
def handle_missing_inplace(dataset):
    dataset['category_name'].fillna(value='missing', inplace=True)
    dataset['general_cat'].fillna(value='missing', inplace=True)
    dataset['subcat_1'].fillna(value='missing', inplace=True)
    dataset['subcat_2'].fillna(value='missing', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='missing', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    pop_category = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    pop_category = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['general_cat'].isin(pop_category), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['category_name'] = dataset['category_name'].astype('category')
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')
    dataset['item_condition_id'] = dataset['item_condition_id'].astype('category')


def main():
    start_time = time.time()

    train = pd.read_table('../input/train.tsv', engine='c')
    test = pd.read_table('../input/test.tsv', engine='c')
    print('[{}] Finished to load data'.format(time.time() - start_time))
    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)

    nrow_train = train.shape[0]
    y = np.log1p(train["price"])
    merge: pd.DataFrame = pd.concat([train, test])
    submission: pd.DataFrame = test[['test_id']]
    cvlist = list(KFold(6, random_state=1).split(train, y))
    #del train
    #del test
    gc.collect()
    
    merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = \
    zip(*merge['category_name'].apply(lambda x: split_cat(x)))
    #merge.drop('category_name', axis=1, inplace=True)
    print('[{}] Split categories completed.'.format(time.time() - start_time))

    handle_missing_inplace(merge)
    print('[{}] Handle missing completed.'.format(time.time() - start_time))

    cutting(merge)
    print('[{}] Cut completed.'.format(time.time() - start_time))

    to_categorical(merge)
    print('[{}] Convert categorical completed'.format(time.time() - start_time))

    cv_name = CountVectorizer(min_df=NAME_MIN_DF)
    cv_name.fit(merge['name'])
    X_name = cv_name.transform(merge['name'])
    X_name_desc = cv_name.transform(merge['item_description'])
    print('[{}] Count vectorize `name` completed.'.format(time.time() - start_time))

    cv = CountVectorizer()
    X_category = cv.fit_transform(merge['category_name'])
    X_category1 = cv.fit_transform(merge['general_cat'])
    X_category2 = cv.fit_transform(merge['subcat_1'])
    X_category3 = cv.fit_transform(merge['subcat_2'])
    print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

    tv = TfidfVectorizer(max_features=MAX_FEATURES_ITEM_DESCRIPTION,
                         ngram_range=(1, 3),
                         stop_words='english')
    X_description = tv.fit_transform(merge['item_description'])
    print('[{}] TFIDF vectorize `item_description` completed.'.format(time.time() - start_time))

    cv = CountVectorizer()
    X_brand = cv.fit_transform(merge['brand_name'])
    X_brand_name = cv.transform(merge['name'])
    print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))

    X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
    print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
    
    print("Get target encodings for stuff")
    enc_1 = TargetEncoder(cols=['brand_name'], func=np.median)
    X_train_brand_mean = cross_val_predict(enc_1, train, y, cv=cvlist, verbose=10, method='transform', n_jobs=1)
    X_test_brand_mean = enc_1.fit(train, y).transform(test)
    X_brand_mean = np.vstack((X_train_brand_mean, X_test_brand_mean))
    #print(X_brand_mean.shape, X_train_brand_mean.shape, X_test_brand_mean.shape)
    
    enc_2 = TargetEncoder(cols=['category_name'], func=np.median)
    X_train_category_mean = cross_val_predict(enc_2, train, y, cv=cvlist, verbose=10, method='transform', n_jobs=-1)
    X_test_category_mean = enc_2.fit(train, y).transform(test)
    X_category_mean = np.vstack((X_train_category_mean, X_test_category_mean))
    
    enc_2_1 = TargetEncoder(cols=['subcat_1'], func=np.median)
    X_train_category1_mean = cross_val_predict(enc_2, train, y, cv=cvlist, verbose=10, method='transform', n_jobs=-1)
    X_test_category1_mean = enc_2.fit(train, y).transform(test)
    X_category1_mean = np.vstack((X_train_category1_mean, X_test_category1_mean))
    
    enc_2_2 = TargetEncoder(cols=['subcat_2'], func=np.median)
    X_train_category2_mean = cross_val_predict(enc_2, train, y, cv=cvlist, verbose=10, method='transform', n_jobs=-1)
    X_test_category2_mean = enc_2.fit(train, y).transform(test)
    X_category2_mean = np.vstack((X_train_category2_mean, X_test_category2_mean))
    
    enc_3 = TargetEncoder(cols=[['brand_name', 'category_name']], func=np.median)
    X_train_brandcat_mean = cross_val_predict(enc_3, train, y, cv=cvlist, verbose=10, method='transform', n_jobs=-1)
    X_test_brandcat_mean = enc_3.fit(train, y).transform(test)
    X_brandcat_mean = np.vstack((X_train_brandcat_mean, X_test_brandcat_mean))
    
    X_train_brandcat_rat = X_train_brandcat_mean/(1 + X_train_category_mean)
    
    X_brandcat_rat = X_brandcat_mean/(1 + X_category_mean)
    X_catbrand_rat = X_brandcat_mean/(1 + X_brand_mean)
    
    enc_4 = TargetEncoder(cols=['brand_name'], func=np.mean)
    X_train_brandvalue = cross_val_predict(enc_4, train, X_train_brandcat_rat, cv=cvlist, verbose=10, method='transform', n_jobs=-1)
    X_test_brandvalue = enc_4.fit(train, X_train_brandcat_rat).transform(test)
    X_brandvalue = np.vstack((X_train_brandvalue, X_test_brandvalue))

    enc_5 = TargetEncoder(cols=['item_condition_id'], func=np.mean)
    X_train_condvalue = cross_val_predict(enc_4, train, X_train_brandcat_rat, cv=cvlist, verbose=10, method='transform', n_jobs=-1)
    X_test_condvalue = enc_4.fit(train, X_train_brandcat_rat).transform(test)
    X_condvalue = np.vstack((X_train_condvalue, X_test_condvalue))
    
    del train, test
    gc.collect()
    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name,
                           X_name_desc,X_category, X_brand_name,
                            X_category_mean,
                            X_category1_mean,
                            X_category2_mean,
                             #X_category_cnt,
                             X_brandcat_mean,
                             #X_brandcat_cnt,
                             X_brandcat_rat,
                             X_catbrand_rat,
                             X_brandvalue,
                             X_condvalue)).tocsr()
    print('[{}] Create sparse merge completed'.format(time.time() - start_time))

    X = sparse_merge[:nrow_train]
    X_test = sparse_merge[nrow_train:]
    del sparse_merge
    
    model = Ridge(solver="sag", fit_intercept=True, random_state=205, max_iter=100, alpha=3)
    preds = cross_val_predict(model, X, y, verbose=10, n_jobs=2, cv=5)
    gc.collect()
        #score = np.mean(scores)
    score = rmse(y, preds)
    print("Ridge CV score :", score)
    
    
    #model.fit(X, y)
    print('[{}] Train ridge completed'.format(time.time() - start_time))
    #predsR = model.predict(X=X_test)
    print('[{}] Predict ridge completed'.format(time.time() - start_time))

    model = SGDRegressor(alpha=0.000001, penalty='l2', l1_ratio=0.15,  learning_rate='invscaling',
      loss='squared_loss', power_t=0.25, random_state=None, shuffle=True, tol=None, verbose=0,
      warm_start=False, average=False, epsilon=0.1, eta0=0.01, fit_intercept=True)
    preds = cross_val_predict(model, X, y, verbose=1, n_jobs=1, cv=5)
    gc.collect()
        #score = np.mean(scores)
    score = rmse(y, preds)
    print("Ridge CV score :", score)
    #model.fit(X, y).sparsify()
    print('[{}] Train sgd completed'.format(time.time() - start_time))
    #predsS = model.predict(X=X_test)
    print('[{}] Predict sgd completed.'.format(time.time() - start_time))

    #train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size = 0.05, random_state = 144) 
    #d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)
    #d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)
    #watchlist = [d_train, d_valid]
    
    params = {
        'learning_rate': 0.75,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 100,
        'verbosity': -1,
        'metric': 'RMSE',
        'nthread': 4
    }

    params2 = {
        'learning_rate': 0.85,
        'application': 'regression',
        'max_depth': 3,
        'num_leaves': 50,
        'verbosity': -1,
        'metric': 'RMSE',
        'nthread': 4
    }

    #model = lgb.train(params, train_set=d_train, num_boost_round=10000, valid_sets=watchlist, \
    #early_stopping_rounds=1000, verbose_eval=1000) 
    #preds = cross_val_predict(model, X, y, verbose=1, n_jobs=1, cv=5)
    #gc.collect()
        #score = np.mean(scores)
    #score = rmse(y, preds)
    #print("Ridge CV score :", score)
    #predsL = model.predict(X_test)
    
    print('[{}] Predict lgb 1 completed.'.format(time.time() - start_time))
    
    #train_X2, valid_X2, train_y2, valid_y2 = train_test_split(X, y, test_size = 0.1, random_state = 101) 
    #d_train2 = lgb.Dataset(train_X2, label=train_y2, max_bin=8192)
    #d_valid2 = lgb.Dataset(valid_X2, label=valid_y2, max_bin=8192)
    #watchlist2 = [d_train2, d_valid2]

    #model = lgb.train(params2, train_set=d_train2, num_boost_round=4000, valid_sets=watchlist2, \
    #arly_stopping_rounds=50, verbose_eval=500) 
    #preds = cross_val_predict(model, X, y, verbose=1, n_jobs=1, cv=5)
    #gc.collect()
        #score = np.mean(scores)
    #score = rmse(y, preds)
    #print("Ridge CV score :", score)
    #predsL2 = model.predict(X_test)

    #print('[{}] Predict lgb 2 completed.'.format(time.time() - start_time))

    # preds = predsR2*0.15 + predsR*0.25 + predsL*0.6
    #preds = predsR*0.25 + predsS*0.1+ predsL*0.45 + predsL2*0.2
    #preds = predsR*0.5 + predsL*0.3 + predsL2*0.2

    #submission['price'] = np.expm1(preds)
    #submission.to_csv("submission_lgbm_sgd_2xridge.csv", index=False)

if __name__ == '__main__':
    main()
