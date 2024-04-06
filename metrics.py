# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 21:53:33 2021

@author: qiang
"""
import torch
import pandas as pd 
#from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy import signal
from scipy import stats



chfont = {'family' : 'SimHei','weight' : 'bold','size' : 12}
font1 = {'family' : 'Arial','weight' : 'bold','size' : 10}
font2 = {'family' : 'Arial','weight' : 'bold','size' : 12}
paw=1.2 #plot axis widith
tks=10
fg=320
colors=['r','g','b','m','c','k','dodgerblue','olive','chocolate']
markers=['v','s','p','d','>','*','<','o','^']

classes=['Malignant','Benign']


headers=np.array(['$\mathbf{T_0}$',
                  '$\mathbf{T_1}$',
                  '$\mathbf{T_2}$',
                  '$\mathbf{T_3}$',
                  '$\mathbf{PP}$',
                  '$\mathbf{AM}$'])


#seldrugs=['DMSO<','Isoproterenol','Norepinephrine']

#,'Sunitinib'
P1_interval=0.012801
P2_interval=0.028802

def plot_sig(pst,pend,h,sig,colors,figsize=8,rotation=0):
    #x = np.ones((2))*xstart[i]
    #y = np.arange(ystart[i],yend[i],(yend[i]-ystart[i])*19/20)
    xys=[]
    xys.append(pst)
    xys.append([pst[0],h])
    xys.append([pend[0],h])
    xys.append(pend)
    xys=np.array(xys)
    plt.plot(xys[:,0],xys[:,1],color="black",linewidth=0.5)

    #x = np.arange(xstart[i],xend[i]+0.1,xend[i]-xstart[i])
    #y = yend[i]+0*x
    #plt.plot(x,y,color="black",linewidth=0.5)
    siglen=len(sig)*10*0.3/(figsize*10)*6# 计算字符占图的比列

    
    x0 = (pst[0]+pend[0])/2
    x0=x0-siglen/2
    
    y0=h+0.005
    plt.annotate(r'%s'%sig, xy=(x0, y0), xycoords='data', xytext=(0, 0),
                 textcoords='offset points', fontsize=9,color=colors,rotation=rotation)


def wilcoxon_signed_rank_test(y1, y2):
	res = stats.wilcoxon(y1, y2)
	print(res)
    
def wilcoxon_rank_sum_test(x, y):
	#res = stats.mannwhitneyu(x ,y)
    res = stats.ranksums(x ,y)
    return res
    #print(res)   

def dc_norm(x):
    tmp=np.array(x)
    j=np.log10(np.max(abs(tmp),0)) 
    return tmp/(10**j)
    

def butterworth_filter(order,x,f,typ,fs):
    if(typ=='low'):
        w1=2*f/fs
        b, a = signal.butter(order, w1, typ)
    elif(typ=='high'):
        w1=2*f/fs
        b, a = signal.butter(order, w1, typ)
    elif(typ=='band'):
        w1=2*f[0]/fs
        w2=2*f[1]/fs
        b, a = signal.butter(order,[w1,w2], typ)
    sf = signal.filtfilt(b,a,x)  
    return sf

def calc_accuracy(output,Y):  
    # get acc_scores during training 
    max_vals, max_indices = torch.max(output,1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    
    #plt.subplot(1,1,1)
    plt.imshow(cm, interpolation='nearest', cmap=cmap,vmin=0,vmax=1)
    #plt.title(title)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center")
    plt.tight_layout()
    
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')


def dtw_distance(ts_a, ts_b, d=lambda x,y: abs(x-y), mww=10000):
    """Computes dtw distance between two time series
    Args:
        ts_a: time series a
        ts_b: time series b
        d: distance function
        mww: max warping window, int, optional (default = infinity)
    Returns: dtw distance
    """
    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = np.ones((M, N)) * 10000
    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
            choices = cost[i-1, j-1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])
    # Return DTW distance given window 
    return cost[-1, -1]
