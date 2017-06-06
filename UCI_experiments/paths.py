# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:07:20 2017

@author: hrs13
"""

shared_path = './'

data_path = shared_path + 'data/'
results_path = shared_path + 'results/'

import os
if not os.path.isdir(results_path):
    os.mkdir(results_path)
    
