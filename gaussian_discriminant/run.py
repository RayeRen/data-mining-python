
# coding: utf-8

# In[1]:

import scipy.io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from gaussian_prob import gaussian_prob
from gaussian_pos_prob import gaussian_pos_prob
from plot_ex1 import plot_ex1
import matplotlib


warnings.filterwarnings("ignore")


# In[19]:

mu0 = np.array(
    [0,1]
)
mu1 = np.array(
    [0,0]
)
sigma0 = np.array(
    [[2,0],
     [0,2]]
)
sigma1 = np.array(
    [[2,0],
     [0,2]]
)
phi = 0.5
fig_title = 'Line'
plot_ex1(mu0, sigma0, mu1, sigma1, phi, fig_title, 0)


# In[37]:

mu0 = np.array(
    [0,1]
)
mu1 = np.array(
    [0,0]
)
sigma0 = np.array(
    [[1,0],
     [0,1]]
)
sigma1 = np.mat(
    [[1,5],
     [-5,1]]
).I
phi = 0.5
fig_title = 'Line (one side)'
plot_ex1(mu0, sigma0, mu1, sigma1, phi, fig_title, 0)


# In[26]:

mu0 = np.array(
    [0,0]
)
mu1 = np.array(
    [0,0]
)
sigma0 = np.array(
    [[3,0],
     [0,1]]
)
sigma1 = np.array(
    [[1,0],
     [0,2]]
)
phi = 0.5
fig_title = 'Hyperbola'
plot_ex1(mu0, sigma0, mu1, sigma1, phi, fig_title, 0)


# In[23]:

mu0 = np.array(
    [0,0]
)
mu1 = np.array(
    [0,1]
)
sigma0 = np.array(
    [[1,0],
     [0,1]]
)
sigma1 = np.array(
    [[2,0],
     [0,1]]
)
phi = 0.5
fig_title = 'Parabalic'
plot_ex1(mu0, sigma0, mu1, sigma1, phi, fig_title, 0)


# In[15]:

mu0 = np.array(
    [0,0]
)
mu1 = np.array(
    [0,0]
)
sigma0 = np.array(
    [[6,0],
     [0,2]]
)
sigma1 = np.array(
    [[2,0],
     [0,2]]
)
phi = 0.5
fig_title = 'Two parallel lines.'
plot_ex1(mu0, sigma0, mu1, sigma1, phi, fig_title, 0)


# In[14]:

mu0 = np.array(
    [0,0]
)
mu1 = np.array(
    [1,0]
)
sigma0 = np.array(
    [[1,0],
     [0,1]]
)
sigma1 = np.array(
    [[0.1,0],
     [0,0.1]]
)
phi = 0.5
fig_title = 'Circle'
plot_ex1(mu0, sigma0, mu1, sigma1, phi, fig_title, 0)


# In[22]:

mu0 = np.array(
    [0,0]
)
mu1 = np.array(
    [1,0]
)
sigma0 = np.array(
    [[1,0],
     [0,1]]
)
sigma1 = np.array(
    [[0.1,0],
     [0,0.5]]
)
phi = 0.5
fig_title = 'Ellipsoid'
plot_ex1(mu0, sigma0, mu1, sigma1, phi, fig_title, 0)


# In[12]:

mu0 = np.array(
    [0,0]
)
mu1 = np.array(
    [0,0]
)
sigma0 = np.array(
    [[1,0],
     [0,1]]
)
sigma1 = np.array(
    [[1,0],
     [0,1]]
)
phi = 0.5
fig_title = 'No boundary'
plot_ex1(mu0, sigma0, mu1, sigma1, phi, fig_title, 0)

