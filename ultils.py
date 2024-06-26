from scipy import optimize
import math
import numpy as np
from scipy import stats
import os
from scipy import signal
filepath='/home/justin/share/figures_materials/single_receptor/'


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def wilcoxon_signed_rank_test(y1, y2):
	res = stats.wilcoxon(y1, y2)
	print(res)
    
def wilcoxon_rank_sum_test(x, y):
	res = stats.mannwhitneyu(x ,y)
	print(res)    
    
#filepath=''
'''Mean square error root'''  
def rmse(y_test, y):  
    return np.sqrt(np.mean((y_test - y) ** 2))  
  
'''The degree of excellence compared to the mean is between [0~1]. 
0 means not as good as the mean. 1 indicates perfect prediction. 
The implementation of this version is based on the scikit-learn official website document.
'''  
def R2(y_test, y_true):  
    return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()  

def func(x,a,b):  
    return a*x+b
    #return a*(np.log(x)/np.log(b))
   
def expfunc(x,a,b):  
    return a*(1+b)**x

def logfunc(x,a,b):  
    return np.log(x/a)/np.log(1+b)
    
def func1(x,a):  
    return a*x    #return a*log(x)+b


def ploy2func(x,a,b,c):  
    return a*x**2+b*x+c    #return a*log(x)+b


def curve_fit1(x,y):
    a=optimize.curve_fit(func1,x,y)[0]
    x1=np.linspace(np.min(x),np.max(x),len(x))
    return [x1,func1(x1,a),func1(x,a),a]

def exp_curve_fiting(x,y):
    a,b=optimize.curve_fit(expfunc,x,y)[0]
    x1=np.linspace(np.min(x),np.max(x),len(x))
    return [x1,expfunc(x1,a,b),expfunc(x,a,b),[a,b]]

def ploy2_curve_fit(x,y):
    a,b,c=optimize.curve_fit(ploy2func,x,y)[0]
    x1=np.linspace(np.min(x),np.max(x),len(x))
    return [x1,ploy2func(x1,a,b,c),ploy2func(x,a,b,c),[a,b,c]]


def log1_curve_fiting(x,y):
    a,b=optimize.curve_fit(logfunc,x,y)[0]
    x1=np.linspace(np.min(x),np.max(x),len(x))
    return [x1,logfunc(x1,a,b),logfunc(x,a,b),[a,b]]

def curve_fit(x,y):
    a,b=optimize.curve_fit(func,x,y)[0]
    x1=np.linspace(np.min(x),np.max(x),len(x))
    return [x1,func(x1,a,b),func(x,a,b),[a,b]]

def linear_curve_fit(x,y):
    a,b=optimize.curve_fit(func,x,y)[0]
    x1=np.linspace(np.min(x),np.max(x),len(x))
    return [x1,func(x1,a,b),func(x,a,b),[a,b]]


def shift_(lst, k):
    return lst[k:] + lst[:k]

def read_data(filename,condition):  
    file=open(filename,'r')  
    buf=[]
    buf.append(-1)
    strbuf=file.readlines()  #
    for i in range(0,len(strbuf)): 
        if len(strbuf[i])==1:  
            buf.append(i)
    file.close()
    buf.append(len(strbuf))
    databuf=np.ones([buf[1]-buf[0]-1,1])*condition[0]
    for i in range(1,len(buf)-1): 
        databuf=np.vstack([databuf,np.ones([buf[i+1]-buf[i]-1,1])*condition[i]])
    return databuf
    #fnew.close()  
    
    
def read_data1(filename):  
    file=open(filename,'r')  
    buf=[]
    buf.append(-1)
    strbuf=file.readlines()  #
    for i in range(0,len(strbuf)): 
        if len(strbuf[i])==1:  
            buf.append(i)
    file.close()
    buf.append(len(strbuf))
    databuf=np.ones([buf[1]-buf[0]-1,1])
    return databuf
    #fnew.close()  
    
 
def read_data2(filename,condition):  
    file=open(filename,'r')  
    buf=[]
    buf.append(-1)
    strbuf=file.readlines()  #
    for i in range(0,len(strbuf)): 
        if len(strbuf[i])==1:  
            buf.append(i)
    file.close()
    buf.append(len(strbuf))
    databuf=np.ones([buf[1]-buf[0]-1,1])*condition[0]
    for i in range(1,len(buf)-1): 
        databuf=np.vstack([databuf,np.ones([buf[i+1]-buf[i]-1,1])*condition[i]])
    return databuf
    #fnew.close()     
    
def text_save(content,filename,mode):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])[1:len(str(content[i]))-1].replace('\n',' ')+'\n')
    file.close()
    
    
def text_read(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename,'r')
    except IOError:
        error = []
        return error
    content = file.readlines()

    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def loadData(path,split=' '):
    #data=list()
    res=list()
    with  open(path,'r') as fileReader:
        lines = fileReader.readlines()  # 读取全部内容
        for line in lines:
            data=[]
            line = line.strip()
            line = line.split(split)#根据数据间的分隔符切割行数据
            #data.append(line[:])
            if(line[0]!=''):
                for i in range(len(line)):
                    if(line[i]!=''):
                        data.append(line[i])
                res.append(data)  
    #data = data.astype(float)
    #np.random.shuffle(data)
    #label=data[:,0]
    #features=data[:,1:]
    #print("data loaded!")
    return res  #features,label-1

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
