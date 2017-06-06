#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:18:59 2017

@author: hrs13
"""


import sys
sys.path.append('../')

import numpy as np

import pickle
import time
import os

from get_data import get_regression_data

from paths import data_path, results_path
from scipy.stats import norm

import geepee.aep_models as aep

from GPflow.sgpr import SGPR, GPRFITC
from GPflow.kernels import RBF

from scipy.cluster.vq import kmeans2

datasets = [
     'boston', #506, 13
     'concrete', #1030, 8
     'energy', #768, 8
     'kin8nm', #8192, 8
     'naval', #11934, 26
     'power', #9568, 4
     'protein', #45730, 9
     'wine_red', #1599, 22
#     'year', #515345, 90
]


identifier = 'test_1'
 
lr = 0.01
mb_size = 100

Models = []

# aegp models
def make_aepdgp_model(L, its, M=100):
    class AEPDGP(aep.SDGPR):
        model_name = 'aepdgp_{}_{}'.format(M, L)
        aep = True
        iterations = its
        def __init__(self, X, Y):
            hidden_size = [2, ] * (L - 1)
            aep.SDGPR.__init__(self, X, Y, M, hidden_size, lik='Gaussian')
    Models.append(AEPDGP)
    
for its in [100,]: #1000, 10000]:
    [make_aepdgp_model(L, its) for L in [1, ]]

## single layer models for comparison 
def make_single_layer_GPflow_models(M=100):
    class FITC(GPRFITC):
        model_name = 'fitc_{}'.format(M)
        aep = False
        def __init__(self, X, Y):
            Z = kmeans2(X, M, minit='points')[0]
            GPRFITC.__init__(self, X, Y, RBF(X.shape[1]), Z)
    Models.append(FITC)
    
    class VFE(SGPR):
        model_name = 'vfe_{}'.format(M)
        aep = False
        def __init__(self, X, Y):
            Z = kmeans2(X, M, minit='points')[0]
            SGPR.__init__(self, X, Y, RBF(X.shape[1]), Z)
    Models.append(VFE)

#make_single_layer_GPflow_models()


def assess_model(model, Xs, Ys):
    n_batches = max(int(Xs.shape[0]/1000.), 1)
    lik, sq_diff = [], []
    for X_batch, Y_batch in zip(np.array_split(Xs, n_batches), 
                                np.array_split(Ys, n_batches)):
        
        # predict mean and var
        mean, var = model.predict_y(X_batch)
       
        # rmse from mean prediction
        sq = ((mean - Y_batch)**2)
        
        l = norm.logpdf(Y_batch, loc=mean, scale=var**0.5)
            
        lik.append(l)
        sq_diff.append(sq)
        
    lik = np.concatenate(lik, 0)
    sq_diff = np.array(np.concatenate(sq_diff, 0), dtype=float)
    return np.average(lik), np.average(sq_diff)**0.5

def do(dataset, split, Model, path):
    # get the data
    X, Y, Xs, Ys = get_regression_data(dataset, split, data_path=data_path)
    
    # init the model 
    model = Model(X, Y)
        
    # train
    if Model.aep is True:
        model.optimise(method='Adam',
                       adam_lr=lr,
                       maxiter=Model.iterations)
    else:
        model.optimize()
        
    # assess 
    lik, rmse = assess_model(model, Xs, Ys)
    
    # save
    with open(path, 'w') as f:
        pickle.dump({'lik':lik, 'rmse':rmse}, f)
    
    print 'lik: {:.4f}, acc: {:.4f}'.format(lik, rmse)
    
    

# randomise to lower chance of intersection
seed = int(1e3*time.time()) % 2**32
np.random.seed(seed)
np.random.shuffle(datasets)
np.random.shuffle(Models)
    
for dataset in datasets:
    if dataset in ['airline', 'year']:
        splits = [0, ]
    else:
        splits = np.arange(20)
    
    np.random.shuffle(splits)

    for split in splits:
        for Model in Models:
            model_name = Model.model_name
            path = '{}{}_{}_{}_{}.p'.format(results_path, 
                               dataset, 
                               model_name, 
                               identifier, 
                               split)
            if not os.path.isfile(path):
                with open(path, 'w') as f:
                    pickle.dump(None, f)

                print '#####################################'
                print path            

                results = do(dataset, split, Model, path)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
             
             
             
             
