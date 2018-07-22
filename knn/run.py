
# coding: utf-8

# In[51]:

import scipy.io as sio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from utils import *
from hack import hack

warnings.filterwarnings("ignore")
get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# # KNN Exp

# In[2]:

X, y = mkdata()
K = [1,10,100]
for k in K:
    plt.figure()
    knn_plot(X, y, k)


# # Hack CAPTCHA

# ##  Generate train data and save

# In[48]:

X = []
y = []
checkcodes_path = "checkcodes"
checkcode_files = [f for f in listdir(checkcodes_path) if isfile(join(checkcodes_path, f))]
for f in checkcode_files:
    file_name = "checkcodes/" + f
    X += extract_image(file_name)
    y += [int(x) for x in f[:5]]
X = np.concatenate(X,axis=1)
y = np.array(y).reshape(1,-1)
sio.savemat('hack_data.mat', {'X':X,'y':y})


# ## Test Hack

# In[66]:

hack(checkcodes_path+"/test_1.jpg")


# In[67]:

hack(checkcodes_path+"/test_2.jpg")


# In[68]:

hack(checkcodes_path+"/test_3.jpg")


# In[69]:

hack(checkcodes_path+"/test_4.jpg")


# In[70]:

hack(checkcodes_path+"/test_5.jpg")


# In[71]:

hack(checkcodes_path+"/test_6.jpg")


# In[72]:

hack(checkcodes_path+"/test_7.jpg")


# In[73]:

hack(checkcodes_path+"/test_8.jpg")


# In[74]:

hack(checkcodes_path+"/test_9.jpg")


# In[75]:

hack(checkcodes_path+"/test_10.jpg")

