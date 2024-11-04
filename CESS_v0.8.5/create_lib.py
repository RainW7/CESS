#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@Env     		:   grizli
@File    		:   ~/emulator/emulator_v0.8/create_lib.py
@Time    		:   2023/08/14 12:52:05
@Author  		:   Run Wen
@Version 		:   0.8
@Contact 		:   wenrun@pmo.ac.cn
@Description	:   To create height and width profile distribution library, change the series values to what you want.
'''

import morphology
import numpy as np 
import pickle
nseries = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1. ,1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2. ,
                   2.5, 3., 3.5, 4., 4.5, 5.]) # 20
reseries = np.round(np.array([0.3, 0.5, 0.7, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 
                              4.5, 5, 5.5, 6, 6.5, 7, 7.4])/0.074)
paseries = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90]) # 10
baseries = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) # 10

widthlib, heightlib = morphology.create_lib(nseries=nseries,reseries=reseries,paseries=paseries,baseries=baseries)

with open('widthlib_20x20x10x10.pkl', 'wb') as f:
    pickle.dump(widthlib, f)

with open('heightlib_20x20x10x10.pkl', 'wb') as f:
    pickle.dump(heightlib, f)
