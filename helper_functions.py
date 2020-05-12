#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np


def save_obj(obj, name):
    pickle.dump(obj,open(name + '.pkl', 'wb'), protocol=4)
    
    
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
    
def commindex(voc, doc):
    '''
    Find index of a word in the vocabulary
    '''
    cluster = np.zeros(len(doc), dtype=np.int64)
    for d in range(len(doc)):
        cluster[d] = voc[doc[d]]
    
    return cluster


def corp2ind(corpus, voc):
    """
    Transfer the words to their indices recorded in the general vocabulary of the dataset
    """
    indCorp = []
    for d in corpus:
        indCorp.append(np.array(commindex(voc, d)))
        
    return indCorp


# @nb.njit()
def weight_inverse(k, sim):
    result = np.zeros(k)
    summ = 0.0
    for i in range(k):
        result[i] += 1.0 / (sim[i]**2+1)
        summ += result[i]
    result = 1 - result/summ
    return result


# @nb.njit(parallel=True)
def knn(sim, k, y):
    classes = len(np.unique(y))
    ordering = sim.argsort()[::-1]    # sort indices of similarities from max to min
    k_near_dists = sim[ordering]      # choose k largest similarities
    votes = np.zeros(classes)
    wts = weight_inverse(k, k_near_dists)   # get weights for similarities
    for j in nb.prange(k):
        idx = ordering[j]
        pred_class = np.int(y[idx])     # take labels of max to min
        votes[pred_class] += wts[j] * 1.0
    return np.argmax(votes)

