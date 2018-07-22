
# coding: utf-8

# In[1]:

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import matplotlib
from os import walk, path
import pickle
from sklearn.metrics import confusion_matrix
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

warnings.filterwarnings("ignore")
get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# ## Preprocess

# In[2]:

all_word = set()
stop = set(stopwords.words('english'))
high_freq_word = set(['a', 'the', 'to']) | stop

all_file = []
for (dirpath, dirnames, filenames) in walk('./'):
    dirpath = dirpath.replace('\\', "/")
    for f in filenames:
        if f.endswith('.txt') and dirpath.startswith('./data/train'):
            all_file.append(path.join(dirpath, f))
            
for f in all_file:
    for x in open(f, encoding='utf-8',errors='ignore').read().lower().split():
        if x.isalpha() and len(x) > 1 and x not in all_word:
            all_word.add(x)
            
all_word = all_word - high_freq_word
id2word = list(all_word)
word2id = dict(list(zip(id2word, list(range(len(id2word))))))


# 
# ## Train
# 

# In[3]:

x_train_ham = np.zeros(len(all_word))
x_train_spam = np.zeros(len(all_word))
ham_total = 0
spam_total = 0

for (dirpath, dirnames, filenames) in walk('./data/train'):
    dirpath = dirpath.replace('\\', "/")
    for f in filenames:
        if f.endswith('.txt'):
            word_appears = np.zeros(len(all_word))
            for x in open(path.join(dirpath, f), encoding='utf-8', errors='ignore').read().lower().split():
                if x.isalpha() and x in all_word and len(x) > 1 and word_appears[word2id[x]]==0:
                    word_appears[word2id[x]] = 1
                    
            if dirpath.startswith('./data/train/ham'):
                ham_total += 1
                x_train_ham += word_appears
            else:
                spam_total += 1
                x_train_spam += word_appears
                
doc_total = ham_total + spam_total
prior = np.array([ham_total / doc_total, spam_total / doc_total])
likelihood_ham = np.log((x_train_ham + 1) / (ham_total + 2))
likelihood_spam = np.log((x_train_spam + 1) / (spam_total + 2))


# ### (4.a) Top10 most indicative of the SPAM class 

# In[10]:

print("Top10 most indicative of the SPAM class:")
for i in np.argsort(likelihood_spam/likelihood_ham)[:10]:
    print(id2word[i])


# ## Test

# In[24]:

labels = []
preds = []

for (dirpath, dirnames, filenames) in walk('./data/test'):
    dirpath = dirpath.replace('\\', "/")
    for f in filenames:
        if f.endswith('.txt'):
            if dirpath.startswith('./data/test/ham'):
                labels.append(1)
            else:
                labels.append(0)
            word_appears = np.zeros(len(all_word))
            for x in open(path.join(dirpath, f), encoding='utf-8', errors='ignore').read().lower().split():
                if x.isalpha() and x in all_word and len(x) > 1 and word_appears[word2id[x]]==0:
                    word_appears[word2id[x]] = 1
                        
            p_ham = (word_appears@likelihood_ham)+np.log(prior[0])
            p_spam = (word_appears@likelihood_spam)+np.log(prior[1])
            if p_ham<p_spam:
                preds.append(0)
            else:
                preds.append(1)

labels = np.array(labels)
preds = np.array(preds)


# ### (4.b) Accuracy

# In[29]:

acc = 1 - (labels^preds).sum()/len(preds)
print("Accuracy: ",acc)


# ### (4.d) Precision and Recall

# In[23]:

tp,fn,fp,tn = list(confusion_matrix(labels,preds).reshape(-1))
precision = tp/(tp+fp)
recall = tp/(tp+fn)
print("Precision: ",precision," Recall: ",recall)

