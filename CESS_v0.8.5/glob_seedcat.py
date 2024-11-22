#!/Users/rain/miniconda3/envs/grizli/bin/python
# -*-coding:utf-8 -*-
'''
@Env     		:   grizli (Python 3.7.11) on Macbook Pro
@File    		:   ~/emulator/CESS_v0.8.5/glob_seedcat.py
@Time    		:   2024/11/04 16:37:35
@Author  		:   Run Wen
@Version 		:   0.8.5
@Contact 		:   wenrun@pmo.ac.cn
@Description	:   Using glob to collect all 'seedcat_*' input file names in terminal
'''

import glob
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python glob.py <seedcat_path>")
    sys.exit(1)

seedcat_path = sys.argv[1]

seedcat_files = glob.glob(seedcat_path + '/seedcat2_*')

for file in seedcat_files:
    print('\'{0}\','.format(os.path.basename(file)))
