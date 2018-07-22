
# coding: utf-8

# In[149]:

import scipy.io as sio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from utils import *
from kmeans import kmeans
from PIL import Image
from vq import vq

warnings.filterwarnings("ignore")
get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# # (a)

# In[4]:

data = sio.loadmat('data/kmeans_data.mat')
X = data['X']


# In[109]:

mindist = 999999
min_params = ()
maxdist = 0
max_params = ()

for i in range(1000):
    idx, ctrs, iter_ctrs = kmeans(X,2)
    dist = np.linalg.norm(X - ctrs[idx])
    if dist<mindist:
        mindist,min_params = dist,(idx, ctrs, iter_ctrs)
    if dist>maxdist:
        maxdist,max_params = dist,(idx, ctrs, iter_ctrs)
        
kmeans_plot(X, *min_params)
kmeans_plot(X, *max_params)


# # (b)

# In[126]:

data = sio.loadmat('data/digit_data.mat')
X = data['X']
X = X.reshape(-1,20,20).transpose(0,2,1).reshape(-1,400)


# In[128]:

show_digit(X[:30])


# In[131]:

for k in [10,20,50]:
    idx, ctrs, _ = kmeans(X,k)
    show_digit(ctrs)


# # vq

# In[172]:

vq(8)


# In[180]:

vq(16)


# In[181]:

vq(32)


# In[192]:

vq(64)

