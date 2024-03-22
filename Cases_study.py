# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:33:25 2024

@author: qiang
"""

import sys
import time
import numpy as np
import pandas as pd 
import warnings
from torch.utils.data.sampler import WeightedRandomSampler
import torch
from torch import optim 
import random
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import metrics as mc
import os
import Model_Metrics as mmc

import scipy

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from statsmodels.stats.libqsturng import psturng
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import Model_Metrics as mmc
import random





path=r"D:\Users\ouyangqq\Desktop\AI_med-img\case_statistics/"

files=os.listdir(path)
print(files)

Malignant_table = np.array(pd.read_excel(path+files[0],sheet_name='统计'))

Benign_table= np.array(pd.read_excel(path+files[1],sheet_name='统计'))

x,y=Malignant_table[:,3],Benign_table[:,2] #Age

# np.median(x)

# x=[57.07168,46.95301,31.86423,38.27486,77.89309,76.78879,33.29809,58.61569,18.26473,62.92256,50.46951,19.14473,22.58552,24.14309]

# y=[8.319966,2.569211,1.306941,8.450002,1.624244,1.887139,1.376355,2.521150,5.940253,1.458392,3.257468,1.574528,2.338976]

res=scipy.stats.ranksums(x, y)

print(res)


'''
headers=np.hstack([ld.BD_tab,np.array('PNS')])
fpath='gdata/analysis_expdata_new.txt'
np.savetxt(fpath,df,fmt='%.03f',delimiter=',')
for i,em in enumerate(ld.BD_tab):
    headers[i]="para"+str(i)

df=pd.read_csv(fpath,header=None,names=headers)
df.to_csv("gdata/ansys_data.csv")
'''

