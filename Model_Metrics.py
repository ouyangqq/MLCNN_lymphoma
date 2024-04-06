# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 20:24:48 2022

@author: qiang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 21:16:20 2021
@author: qiang
"""
import sys
sys.path.append('../')
import time
import numpy as np
import pandas as pd 
import warnings
from torch.utils.data.sampler import WeightedRandomSampler
from torch import optim 
import torch
import random
import torch.utils.data
from torch.utils.data.sampler import SubsetRandomSampler
import random
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
import ultils as alt
from sklearn.metrics import roc_curve, auc
import metrics as mc
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve


def plot_all_weight(x,ys,sfname=''):
    plt.figure(figsize=(14,8))
    plt.subplots_adjust(hspace=0.2) 
    plt.subplots_adjust(wspace=0.1) 
    for no in range(10):
        ax = plt.subplot(5,2,no+1)
        ax.spines['bottom'].set_linewidth(1);##设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1);##设置左边坐标轴的粗细
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        y=ys[no]
        num=len(y[y>0.01])
        plt.bar(np.arange(0,len(x)), y,label="Indi_num:"+str(num))
        plt.xticks(np.arange(0,len(x)),x,rotation=90,fontsize=6,color='w')
        if(no==8 )| (no==9):plt.xticks(np.arange(0,len(x)),x,rotation=90,fontsize=6,color='k')
        plt.ylim(0,1)
        plt.xlim(-1,len(x))
        plt.legend(loc="upper right",fontsize=10)
    plt.savefig('saved_figs/'+sfname+'weight_all.png',bbox_inches='tight', dpi=300) 
    

def plot_inds_PNS(x,ys,yeorr,sfname='',labels=''):
    plt.figure(figsize=(9,10))
    plt.subplots_adjust(hspace=0.3) 
    plt.subplots_adjust(wspace=0.3) 
    for no in range(len(ys)):
        ax = plt.subplot(8,6,no+1)
        ax.spines['bottom'].set_linewidth(1);##设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1);##设置左边坐标轴的粗细
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        y=ys[no]
        num=len(y[y>0.01])
        plt.bar(x, y,yerr=yeorr[no],label=labels[no],ec='k',capsize=2)
        plt.xticks(np.arange(0,len(x)),x,rotation=90,fontsize=6,color='w')
        if(no==len(ys)-2 )| (no==len(ys)-1):plt.xticks(np.arange(0,len(x)),x,rotation=90,fontsize=10,color='k')
        # plt.ylim(0,np.round(y.max(),1)+0.22)
        plt.xlim(-1,len(x))
        # plt.legend(loc="upper right",fontsize=8)
        plt.title(labels[no],fontsize=10)
    plt.savefig('saved_figs/'+sfname+'weight_all.png',bbox_inches='tight', dpi=300) 
    
def plot_single_weight(x,y,sfname=''):
    fig = plt.figure(figsize=(10,2))
    ax = fig.add_subplot(111)
    ax.spines['bottom'].set_linewidth(1);##设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1);##设置左边坐标轴的粗细
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    plt.bar(np.arange(0,len(x)), y)
    plt.xticks(np.arange(0,len(x)),x,rotation=90,fontsize=7)
    plt.ylim(0,1)
    plt.xlim(-1,len(x))
    
    plt.savefig('saved_figs/'+sfname+'weight.png',bbox_inches='tight', dpi=300) 


def plot_loss_accuracy(lossaccbf,sfname=''):
    fig = plt.figure(figsize=(3.4,3))
    ax1 = fig.add_subplot(111)
    ax1.spines['top'].set_linewidth(0);##设置底部坐标轴的粗细
    ax1.spines['bottom'].set_linewidth(1);##设置底部坐标轴的粗细
    ax1.spines['left'].set_linewidth(1);##设置左边坐标轴的粗细
    ax1.spines['right'].set_linewidth(1);##设置左边坐标轴的粗细
    xs=np.arange(0,len(lossaccbf))
    plt.ylim(0,1)
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0],fontsize=10)#,fontweight='bold'
    plt.xticks(fontsize=10)
    st=0#int(len(lossaccbf)/50)
    sp=1
    end=30000
    ax1.plot(xs[st:end:sp],lossaccbf[st:end:sp,0],'k-',marker='d',markersize=1,markerfacecolor='none',linewidth=0.2,label=' \n\n\nTrain loss     \n\n\n')
    ax1.set_xlabel('epochs',mc.font1)
    ax1.set_ylabel('Training loss',mc.font1)  # 可以使用中文，但需要导入一些库即字体
    #plt.title('ROC Curve for class '+ str(class_id))
    ax1.legend(loc='upper right',edgecolor='w')
    
    ax2 = ax1.twinx() 
    ax2.spines['top'].set_linewidth(0);##设置底部坐标轴的粗细
    ax2.spines['bottom'].set_linewidth(1);##设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(1);##设置左边坐标轴的粗细
    ax2.spines['right'].set_linewidth(1);##设置左边坐标轴的粗细
    ax2.plot(xs[st:end:sp],lossaccbf[st:end:sp,1],'b-',marker='o',markersize=1,linewidth=0.2,markerfacecolor='none',label='Train accuracy')
    ax2.plot(xs[st:end:sp],lossaccbf[st:end:sp,2],'m-',marker='o',markersize=1,linewidth=0.2,markerfacecolor='none',label='Test accuracy')
    
    plt.xticks(fontsize=10)#,fontweight='bold'
    plt.yticks([0,0.2,0.4,0.6,0.8,1],fontsize=10)
    plt.ylim(0.5,1)
    #plt.xlabel('epochs',mc.font1)
    ax2.set_ylabel('Accuracy',mc.font1) # 可以使用中文，但需要导入一些库即字体
    #plt.title('ROC Curve for class '+ str(c
    ax2.legend(loc='upper right' ,edgecolor='w')#,bbox_to_anchor=(1.02,0.54)
    
    
    # print('loss:',lossaccbf[-1,0],
    #       'Trianing acc:',lossaccbf[-1,1],
    #       'Testing acc:',lossaccbf[-1,2])
    plt.savefig('saved_figs/'+sfname+'lossacc.png',bbox_inches='tight', dpi=300) 

def plot_loss(lossaccbf,sfname=''):
    fig = plt.figure(figsize=(3.4,3))
    ax1 = fig.add_subplot(111)
    ax1.spines['top'].set_linewidth(0);##设置底部坐标轴的粗细
    ax1.spines['bottom'].set_linewidth(1);##设置底部坐标轴的粗细
    ax1.spines['left'].set_linewidth(1);##设置左边坐标轴的粗细
    ax1.spines['right'].set_linewidth(0);##设置左边坐标轴的粗细
    xs=np.arange(0,len(lossaccbf))
    plt.ylim(0,0.5)
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5],fontsize=10)#,fontweight='bold'
    plt.xticks(fontsize=10)
    st=0#int(len(lossaccbf)/50)
    sp=1
    end=30000
    ax1.plot(xs[st:end:sp],lossaccbf[st:end:sp,0],'k-',marker='d',markersize=1,markerfacecolor='none',linewidth=0.2,label='Train loss')
    
    ax1.plot(xs[st:end:sp],lossaccbf[st:end:sp,1],'b-',marker='o',markersize=1,linewidth=0.2,markerfacecolor='none',label='Test loss')
    
    ax1.set_xlabel('epochs',mc.font1)
    ax1.set_ylabel('Loss',mc.font1)  # 可以使用中文，但需要导入一些库即字体
    #plt.title('ROC Curve for class '+ str(class_id))
    ax1.legend(loc='upper right',edgecolor='w')

    
    # print('loss:',lossaccbf[-1,0],
    #       'Trianing acc:',lossaccbf[-1,1],
    #       'Testing acc:',lossaccbf[-1,2])
    plt.savefig('saved_figs/'+sfname+'lossacc.png',bbox_inches='tight', dpi=300) 

def plot_loss1(lossaccbf,sfname=''):
    fig = plt.figure(figsize=(3.4,3))
    ax1 = fig.add_subplot(111)
    ax1.spines['top'].set_linewidth(0);##设置底部坐标轴的粗细
    ax1.spines['bottom'].set_linewidth(1);##设置底部坐标轴的粗细
    ax1.spines['left'].set_linewidth(1);##设置左边坐标轴的粗细
    ax1.spines['right'].set_linewidth(0);##设置左边坐标轴的粗细
    xs=np.arange(0,len(lossaccbf))
    plt.ylim(0,0.1)
    plt.yticks([0,0.02,0.04,0.06,0.08,0.1],fontsize=10)#,fontweight='bold'
    plt.xticks(fontsize=10)
    st=0#int(len(lossaccbf)/50)
    sp=1
    end=30000
    ax1.plot(xs[st:end:sp],lossaccbf[st:end:sp,0],'k-',marker='d',markersize=1,markerfacecolor='none',linewidth=0.2,label='Train loss')
    
    ax1.plot(xs[st:end:sp],lossaccbf[st:end:sp,1],'b-',marker='o',markersize=1,linewidth=0.2,markerfacecolor='none',label='Test loss')
    
    ax1.set_xlabel('epochs',mc.font1)
    ax1.set_ylabel('Loss',mc.font1)  # 可以使用中文，但需要导入一些库即字体
    #plt.title('ROC Curve for class '+ str(class_id))
    ax1.legend(loc='upper right',edgecolor='w')

    
    # print('loss:',lossaccbf[-1,0],
    #       'Trianing acc:',lossaccbf[-1,1],
    #       'Testing acc:',lossaccbf[-1,2])
    plt.savefig('saved_figs/'+sfname+'lossacc.png',bbox_inches='tight', dpi=300) 


def plot_calibration_curve(final_y,YY,sfname=''):
    
    
    class_Label=torch.max(YY,1)[1].cpu().detach().numpy()
    class_pred=F.softmax(final_y, dim=1).cpu().detach().numpy()
    #tmp=class_pred[:,1]-class_pred[:,0]
    #class_pred=(tmp-tmp.min())/(tmp.max()-tmp.min())
    
    prob_true, prob_pred = calibration_curve(class_Label, class_pred[:,1], n_bins=10)
    plt.figure(figsize=(5,4))
    ax=plt.subplot(1,1,1)
    #ax.spines['top'].set_linewidth(2);##设置底部坐标轴的粗细
    ax.spines['bottom'].set_linewidth(1);##设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1);##设置左边坐标轴的粗细
    #ax.spines['right'].set_linewidth(2);##设置左边坐标轴的粗细
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    plt.plot(prob_pred,prob_true,'k-',label='Truth',markerfacecolor='none',marker='o',markersize=4)
    #plt.plot(prob_pred,'r-',label='Model',markerfacecolor='none',marker='d',markersize=4)
    plt.ylim(0,1)
    plt.xlim(0,1)#len(prob_pred)-1
    plt.legend()
    plt.savefig('saved_figs/'+sfname+'_calibration_curve_CNN_classifier.png',bbox_inches='tight', dpi=300) 
    return prob_true,prob_pred
    
    
def plot_roc(Y_test,final_y,labels,sfname=''):
    fig = plt.figure(figsize=(4,3.4))
    plt.subplots_adjust(hspace=0.3) 
    plt.subplots_adjust(wspace=0.3) 
    N=Y_test.shape[0]
    res=[]
    
    max_vals,Truths= torch.max(Y_test[:N,:len(labels)],1)
    y_label=Truths.cpu().detach().numpy()+1
    
    A_tmp=final_y.cpu().detach().numpy()[:,0:len(labels)]
    
    #A_tmp=F.softmax(final_y, dim=1).cpu().detach().numpy()[:,0:len(labels)]
    
    
    # A_max,A_min=np.max(A_tmp,1).reshape(N,1),np.min(A_tmp,1).reshape(N,1)
    # A_tmp=(A_tmp-A_min)/(A_max-A_min)
    # A_tmp=A_tmp/np.sum(A_tmp,1).reshape(N,1)
    
    
    for idc in range(len(labels)):
        ax=plt.subplot(1,1,1)
        #ax.spines['top'].set_linewidth(2);##设置底部坐标轴的粗细
        ax.spines['bottom'].set_linewidth(1);##设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1);##设置左边坐标轴的粗细
        #ax.spines['right'].set_linewidth(2);##设置左边坐标轴的粗细
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        #ax.spines['top'].set_color('None')
        #ax.spines['right'].set_color('None')
        

        
        y_pre=A_tmp[:,idc]
        
        res.append(y_pre)
        fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=idc+1)

        # print(fpr,tpr)
        roc_auc=auc(fpr,tpr)
        title=labels[idc]
        xm=1
        sel=fpr<=xm
        #print(sel)
        Auc=' (AUC = {0:.4f})'.format(roc_auc)
        plt.plot(fpr[sel], tpr[sel], '-',linewidth=0.4,color=mc.colors[idc],markerfacecolor='none',marker='o',markersize=4,
                 label=title+Auc)
        plt.xlim([0, xm])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([0, 1])
        plt.xticks(fontsize=10)#,fontweight='bold'
        plt.yticks(fontsize=10)
        plt.xlabel('False Positive Rate',mc.font1)
        plt.ylabel('True Positive Rate',mc.font1)  # 可以使用中文，但需要导入一些库即字体
        #plt.title('ROC Curve for class '+ str(idc))
        plt.legend(loc="lower right",fontsize=10)#loc="lower right" bbox_to_anchor=(0.1, 0.60)
    
    plt.show()
    plt.savefig('saved_figs/'+sfname+'ROC_curve.png',bbox_inches='tight', dpi=300) 
    
    return A_tmp



def plot_R2(Y_test,final_y,seldrugs,am=50,selc=-2,sfname='',dgns=[2,8]):
    plt.figure(figsize=(12,1.8))
    plt.subplots_adjust(hspace=0.3) 
    plt.subplots_adjust(wspace=0.35) 
    pltc=0
    for idc in range(dgns[0],dgns[1]):
        pltc=pltc+1
        ax=plt.subplot(1,6,pltc)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        ax.spines['bottom'].set_linewidth(2);##设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(2);##设置左边坐标轴的粗细
        
        yls=Y_test.cpu().detach().numpy()
        pre_yls=final_y.cpu().detach().numpy()
        sels=(pre_yls[:,selc]<am*1.1)
        yls=yls[sels,:]
        pre_yls=pre_yls[sels,:]
        
        ylabel_c=yls[yls[:,idc]==1,selc]
        
        ypre_c=pre_yls[yls[:,idc]==1,selc]
    
        
        #Correlation of concentraction predictor
        [R2_value,pval]=stats.pearsonr(ypre_c,ylabel_c)
        print([R2_value,pval])
        R2_value=round(R2_value,3)
        
        res=alt.linear_curve_fit(ylabel_c,ypre_c)
        
        
        plt.scatter(ylabel_c,ypre_c,s=5,color='w',edgecolors=mc.colors[idc])
        
        #plt.plot(res[0],res[1],color='k',linewidth=3,label='$\mathrm{R}^{2}$='+str(R2_value))
        #mc.colors[idc]
        plt.plot(ylabel_c,ylabel_c,color='k',linewidth=1,label='$\mathrm{R}^{2}$='+str(R2_value))
        
        #plt.yticks([0,4,8,12],fontweight='bold')
        #plt.xticks([0,4,8,12],fontweight='bold')
        
        #if(idc==4):plt.xlabel('                              Normalized logarithmic Reference DC',mc.font1)
        #if(idc==2):plt.ylabel('Normalized logarithmic\n prediected DC',mc.font1) # 可以使用中文，但需要导入一些库即字体
        

        
        if(seldrugs[idc]=='Acetycholin'): title='Acetylcholine' 
        else:title=seldrugs[idc]
        
        plt.title(title.lower(),fontsize=12,fontweight='bold')
        
        #plt.legend(loc='upper left',edgecolor='w')
        plt.text(ylabel_c.max()/2,0,'$\mathrm{R}^{2}$='+str(R2_value))
        
    plt.savefig('saved_figs/'+sfname+'correlation.png',bbox_inches='tight', dpi=300)     
    
    '''
    max_vals,Truths= torch.max(Y_test[:N,:],1)
    Ps=final_y.reshape(final_y.shape[0],final_y.shape[-1])
    max_vals,Preds= torch.max(Ps,1)
    import metrics as mc
    from sklearn.metrics import confusion_matrix
    print('Overall Accuracy',mc.calc_accuracy(Ps,Truths))
    classes={0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q',5: 'S',
             6: 'N', 7: 'S'}
    #classes={0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q',5: 'S',
    #         6: 'N', 7: 'S', 8: 'V', 9: 'F', 10: 'Q',11: 'S',
    #         12: 'N', 13: 'S', 14: 'V', 15: 'F', 16: 'Q',17: 'S',
    #         18: 'N', 19: 'S', 20: 'V', 21: 'F', 22: 'Q',23: 'S',
    #         24: 'N', 25: 'S', 26: 'V', 27: 'F', 28: 'Q',29: 'S',
    #         30: 'N', 31: 'S', 32: 'V', 33: 'F', 34: 'Q',35: 'S'}
    
    cm = confusion_matrix(y_true=Truths.cpu().detach().numpy(), y_pred=Preds.cpu().detach().numpy())
    mc.plot_confusion_matrix(cm=cm,normalize=True,classes=classes)
    plt.savefig('saved_figs/MLP_confused_matrix.png',bbox_inches='tight', dpi=300) 
    
    '''
def plot_concentration_boxplot(Y_test,final_y,seldrugs,am=10,selc=-2,sfname='',dgns=[2,8]):
    cmin,cmax=0.00032,2# um
    fig = plt.figure(figsize=(6,6))
    plt.subplots_adjust(hspace=0.55) 
    plt.subplots_adjust(wspace=0.3) 
    pltc=0
    for idc in range(dgns[0],dgns[1]):
        pltc=pltc+1
        ax=plt.subplot(2,3,pltc)
        ax.spines['bottom'].set_linewidth(2);##设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(2);##设置左边坐标轴的粗细
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        
        yls=Y_test.cpu().detach().numpy()
        pre_yls=final_y.cpu().detach().numpy()
        
        (np.log10(cmax)-np.log10(cmin))*cmax+np.log10(cmin)
        
        sels=(pre_yls[:,selc]<am*1.1)
        yls=yls[sels,:]
        pre_yls=pre_yls[sels,:]
        
        #print(pre_yls.shape)
        
        ylabel_c=yls[yls[:,idc]==1,selc]
        dgcvs=np.float32(np.unique(ylabel_c))
        dgcvs.sort()
        dgdts=dgcvs/am
        dgdts=(np.log10(cmax)-np.log10(cmin))*dgdts+np.log10(cmin)
        dglabels=np.array(dgdts)
        
        dgdts=10**dgdts
        dglbs=[]
        print(dgdts)
        for j,dc in enumerate(dgdts):
            
            if(dc>0.1):dglbs.append('{0:.2f}'.format(dc)+'$\mathbf{\mu}$M')
            else:dglbs.append('{0:.2f} nM'.format(dc*1000))
        #print(dglbs)
        ypre_c=pre_yls[yls[:,idc]==1,selc]
        #j*np.ones(len(dat)),
        bf=[]
        for j,dc in enumerate(dgcvs):
            dat=ypre_c[ylabel_c==dc]/am
            dat=(np.log10(cmax)-np.log10(cmin))*dat+np.log10(cmin)
            bf.append(dat)
            plt.plot([0,1,2,3,4],dglabels[j]*np.ones(5),'k--',alpha=0.3)
        
        for m in range(1,4):
            y1=bf[m]
            y2=bf[m-1]
            
            res=mc.wilcoxon_rank_sum_test(y1, y2)
            pval=np.array(res)[1]
            print(len(y1),pval)

            yst=np.median(y2)#+np.std(y2)
            yend=np.median(y1)#+np.std(y1)
            pst=[m+0.3,yst]
            pend=[m+1-0.3,yend]
            hm=yend+0.5

            if(pval>0.1):sig='ns'
            elif(pval>0.01)&(pval<=0.1):sig='*'
            elif(pval>0.001)&(pval<=0.01):sig='**'
            else:sig='***'
            mc.plot_sig(pst,pend,hm,sig,'k')
            #print(np.array(res)[1])
        
        plt.boxplot(x=np.array(bf),whis=1.5)
        plt.rcParams['boxplot.flierprops.markersize'] =4 # 默认为6
        #plt.yticks(np.round(np.linspace(-3,0.3,5),1))
        #plt.plot(dc*np.ones())
        dglbs.append(' ')
        plt.xticks([1,2,3,4,5],dglbs,rotation=45,fontweight='bold')
        plt.yticks([-3,-2,-1,0,1],fontweight='bold')
        #plt.xticks([0,0.5,1])
        '$\mathrm{log}_{10}$'
        if(idc==6):plt.xlabel('Reference DC',mc.font1)
        if(idc==5)|(idc==1):plt.ylabel('Logarithmic DC',mc.font1)  # 可以使用中文，但需要导入一些库即字体
        title=seldrugs[idc]
    
        
        if(seldrugs[idc]=='Acetycholin'): title='Acetylcholine'
        else:title=seldrugs[idc]
        
        plt.title(title.lower(),fontsize=12,fontweight='bold')
        #if((idc==1)&(idc==4)): plt.ylabel('Logarithmic DC')
        #plt.legend(loc='upper left',edgecolor='w')
        #plt.text(ylabel_c.max()*0.35,ypre_c.min()+1,'$\mathrm{R}^{2}$=')    
    plt.savefig('saved_figs/'+sfname+'concentration_boxplot.png',bbox_inches='tight', dpi=300)     


def plot_scatter(nrow,ncol,no,data,sfname='ree'):
    fig = plt.figure(figsize=(12,8))
    plt.subplots_adjust(hspace=0.55) 
    plt.subplots_adjust(wspace=0.1) 
    ax=plt.subplot(nrow,ncol,no)
    ax.spines['bottom'].set_linewidth(2);##设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);##设置左边坐标轴的粗细
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    plt.ylim(0,1)
    plt.xlim(0,10)
    plt.yticks(fontweight='bold')
    plt.xticks(fontweight='bold')
    #plt.xticks(range(0,len(bf)),ylabel_c,rotation=45,fontweight='bold')
    #plt.xticks([1,2,3,4,5],dglbs,rotation=45,fontweight='bold')
    #plt.yticks([-3,-2,-1,0,1],fontweight='bold')
    #plt.xticks([0,0.5,1])
    #'$\mathrm{log}_{10}$'
    #if(idc==6):plt.xlabel('Reference DC',mc.font1)
    #if(idc==5)|(idc==1):plt.ylabel('Logarithmic DC',mc.font1)  # 可以使用中文，但需要导入一些库即字体
    title=''
    
    plt.title(title.lower(),fontsize=12,fontweight='bold')
    #if((idc==1)&(idc==4)): plt.ylabel('Logarithmic DC')
      #plt.legend(loc='upper left',edgecolor='w')
    #plt.text(ylabel_c.max()*0.35,ypre_c.min()+1,'$\mathrm{R}^{2}$=')    
    
    for idc in range(len(data)):
        plt.scatter(data[0],data[1],s=5,color='w',edgecolors=mc.colors[idc])
        #plt.boxplot(x=np.array(bf),whis=1.5)
    plt.rcParams['boxplot.flierprops.markersize'] =4 # 默认为6

    plt.savefig('saved_figs/'+sfname+'data_plot.svg',bbox_inches='tight', dpi=300) 


def data_boxplot(Y_test,final_y,seldrugs,am=10,selc=-2,sfname='',dgns=[2,8]):
    cmin,cmax=0.00032,2# um
    fig = plt.figure(figsize=(12,8))
    plt.subplots_adjust(hspace=0.55) 
    plt.subplots_adjust(wspace=0.1) 
    pltc=0
    for idc in range(dgns[0],dgns[1]):
        pltc=pltc+1
        ax=plt.subplot(3,2,pltc)
        ax.spines['bottom'].set_linewidth(1);##设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1);##设置左边坐标轴的粗细
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        
        yls=Y_test.cpu().detach().numpy()
        pre_yls=final_y.cpu().detach().numpy()
        
        (np.log10(cmax)-np.log10(cmin))*cmax+np.log10(cmin)
        
        sels=(pre_yls[:,selc]<am*1.1)
        yls=yls[sels,:]
        pre_yls=pre_yls[sels,:]
        
        #print(pre_yls.shape)
        
        ylabel_c=yls[yls[:,idc]==1,selc]
        dgcvs=np.float32(np.unique(ylabel_c))
        dgcvs.sort()
        if(len(dgcvs)>20):dgcvs=dgcvs[0:21]
        
        dgdts=dgcvs/am
        #dgdts=(np.log10(cmax)-np.log10(cmin))*dgdts+np.log10(cmin)
        dglabels=np.array(dgdts)
        
        dgdts=10**dgdts
        dglbs=[]
        print(dgdts)
        for j,dc in enumerate(dgdts):
            
            if(dc>0.1):dglbs.append('{0:.2f}'.format(dc)+'$\mathbf{\mu}$M')
            else:dglbs.append('{0:.2f} nM'.format(dc*1000))
        #print(dglbs)
        ypre_c=pre_yls[yls[:,idc]==1,selc]
        #j*np.ones(len(dat)),
        bf=[]
        for j,dc in enumerate(dgcvs):
            dat=ypre_c[ylabel_c==dc]/am
            #dat=(np.log10(cmax)-np.log10(cmin))*dat+np.log10(cmin)
            bf.append(dat)
            #plt.plot([0,1,2,3,4],dglabels[j]*np.ones(5),'k--',alpha=0.3)
        
        for m in range(1,len(bf)):
            y1=bf[m]
            y2=bf[m-1]
            
            res=mc.wilcoxon_rank_sum_test(y1, y2)
            pval=np.array(res)[1]
            print(len(y1),pval)

            yst=np.median(y2)#+np.std(y2)
            yend=np.median(y1)#+np.std(y1)
            pst=[m+0.3,yst]
            pend=[m+1-0.3,yend]
            hm=yend+0.5

            if(pval>0.1):sig='ns'
            elif(pval>0.01)&(pval<=0.1):sig='*'
            elif(pval>0.001)&(pval<=0.01):sig='**'
            else:sig='***'
            #mc.plot_sig(pst,pend,hm,sig,'k')
            #print(np.array(res)[1])
        plt.scatter(ylabel_c,ypre_c/am,s=5,color='w',edgecolors=mc.colors[idc])
        #plt.boxplot(x=np.array(bf),whis=1.5)
        plt.rcParams['boxplot.flierprops.markersize'] =4 # 默认为6
        #plt.yticks(np.round(np.linspace(-3,0.3,5),1))
        #plt.plot(dc*np.ones())
        dglbs.append(' ')
        
        #plt.xticks([0,2,4,6,8,10],fontweight='bold')
        plt.ylim(0,1)
        plt.xlim(0,10)
        plt.yticks(fontweight='bold')
        plt.xticks(fontweight='bold')
        #plt.xticks(range(0,len(bf)),ylabel_c,rotation=45,fontweight='bold')
        
        #plt.xticks([1,2,3,4,5],dglbs,rotation=45,fontweight='bold')
        #plt.yticks([-3,-2,-1,0,1],fontweight='bold')
        #plt.xticks([0,0.5,1])
        '$\mathrm{log}_{10}$'
        #if(idc==6):plt.xlabel('Reference DC',mc.font1)
        #if(idc==5)|(idc==1):plt.ylabel('Logarithmic DC',mc.font1)  # 可以使用中文，但需要导入一些库即字体
        title=seldrugs[idc]
    
        
        if(seldrugs[idc]=='Acetycholin'): title='Acetylcholine'
        else:title=seldrugs[idc]
        
        plt.title(title.lower(),fontsize=12,fontweight='bold')
        #if((idc==1)&(idc==4)): plt.ylabel('Logarithmic DC')
        #plt.legend(loc='upper left',edgecolor='w')
        #plt.text(ylabel_c.max()*0.35,ypre_c.min()+1,'$\mathrm{R}^{2}$=')    
    plt.savefig('saved_figs/'+sfname+'data_boxplot'+str(selc)+'.svg',bbox_inches='tight', dpi=300)   

def plot_confusion_matrix(org_pred,orgLabel,sfname='',classes=''):  
    max_vals,Truths= torch.max(orgLabel,1)
    Ps=org_pred.reshape(org_pred.shape[0],org_pred.shape[-1])
    max_vals,Preds= torch.max(Ps,1)
    
    plt.figure(figsize=(3.5,3.5))
    cm = confusion_matrix(y_true=Truths.cpu().detach().numpy(), y_pred=Preds.cpu().detach().numpy())
    mc.plot_confusion_matrix(cm=cm,normalize=True,classes=classes)
    
    buf=[]
    for i in range(0,cm.shape[0]):
        buf.append(cm[i,i])
    acc=np.round(np.sum(buf)/np.sum(cm),3)
    print('#############',acc)
    plt.title("Accuracy="+str(acc),mc.font1)
    # plt.colorbar()
    plt.savefig('saved_figs/'+sfname+'confused_matrix.png',bbox_inches='tight', dpi=300) 
    


def plot_data():
    plt.figure(figsize=(3.4,3))
    plt.subplots_adjust(hspace=0.3) 
    plt.subplots_adjust(wspace=0.3) 
    for j,dc in enumerate([]):
        ax=plt.subplot(1,1,1)
        #plt.scatter()
        #plt.boxplot()
        #plt.errorbar()
        #plt.plot
        ax.spines['top'].set_color('None')
        ax.spines['right'].set_color('None')
        ax.set_xlim(0,10)
        ax.set_ylim(0,1)
        
        ax.set_xticks(fontweight='bold')
        ax.set_yticks(fontweight='bold')