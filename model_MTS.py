#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from numba import jit, njit
from numba.types import float64, int64
import numpy as np



@jit(float64[:](float64[:], int64, float64[:]),nopython=True)
def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)


@nb.njit()
def cross_similarity(similarity_matrix, row_weights, col_weights, eps, P, neg1, neg2, idx1, idx2):
    """
    This function takes all prepared parameters to make a final calculation of the final similarity value.
    
    _____Inputs_____
    
    similarity_matrix:   pairwise similarity values with rows representing words from text1, columns from text2
    row_weights:   weights of words in text1
    col_weights:   weights of words in text2
    eps:   all similarity value relations less than this epsilon threshold will be ignored
    P:   regularization parameter (can be set up tp 0)
    neg1:  g(w_text2) - binaries for text1, where 1 means we keep this word for further evaluations, 0 means 
           its out of boundaries of 1 sigma and it is not significant to contribute to the true similarity value
    neg2:  g(w_text1)
    idx1:  m_D - distribution of maximum similarity values for each word in text1
    idx2:  m_T - for text2
    
    _____Output_____
    
    connection_value:  final similarity value of text1 and text2
    """
    n_rows = len(row_weights)
    n_cols = len(col_weights)
    assert(similarity_matrix.shape == (n_rows,n_cols))
    true_connection_value = 0
    eval_connection_value = 0

    # evaluation of true relations (for more details refer to the paper)
    for i in nb.prange(n_rows):
        true_connection_value += (row_weights[i]*col_weights[idx1[i]])*(similarity_matrix[i,idx1[i]])*neg1[i]
        for j in range(n_cols):
            if i == 0:
                true_connection_value += (row_weights[idx2[j]]*col_weights[j])*(similarity_matrix[idx2[j],j])*neg2[j]

            # evaluation of estimated relations
            if similarity_matrix[i,j] >= eps:
                eval_connection_value += (row_weights[i]*col_weights[j])*(P + similarity_matrix[i,j])

    if true_connection_value > 0:
        connection_value = eval_connection_value/true_connection_value
    else:
        connection_value = 0

    return connection_value
    
    
@nb.njit()     
def sim(text1, text2, w_1, w_2, D, eps, P, sigm_coef):
    """
    This function makes all necessary calculations for the estimation of true relation.
    
    _____Inputs_____
    
    text1, text2:   two texts to compare represented in the index form obtained from corp2ind() function,
                    where indices recorded from all words in vocabulary
    w_1:   weights of words in text1
    w_2:   weights of words in text2
    D:   precalculated pairwise distance matrix of all words in vocabulary
    eps:  all similarity value relations less than this epsilon threshold will be ignored
    P:   regularization parameter (can be set up tp 0)
    sigm_coef:   coefficient of sigma, if 1, then words out of the bounds of 1 sigma will be rejected,
                but the bounds can be extended or reduced if preferred. By default, should be equal to 1
                
    _____Output_____
    
    sim:   final similarity value of text1 and text2
    """
    M = np.zeros((len(text1),len(text2)))
    
    # Obtain the similarity matrix for the related words, where rows represent  
    # words of text1 and columns are for words of text2
    for d in range(len(text1)):
        for t in range(len(text2)):
            M[d,t] = 1 - D[text1[d],text2[t]]
            
    max1 = np.array([np.amax(M[m,:]) for m in range(M.shape[0])])
    # m_D, max value from each word (index) in text1 to words (values in list) in text2
    indmax1 = [np.where(M[m,:] == max1[m])[0][0] for m in range(M.shape[0])] 
    max2 = np.array([np.amax(M[:,n]) for n in range(M.shape[1])])
    # m_T
    indmax2 = [np.where(M[:,n] == max2[n])[0][0] for n in range(M.shape[1])]
    
    # find which max similarities are out of one sigma
    # decision_D
    find1 = (max1 - np.mean(max1) + sigm_coef * np.std(max1))
    # decision_T
    find2 = (max2 - np.mean(max2) + sigm_coef * np.std(max2))
    
    # assign 1 to words higher than -sigm_coef*sigma, and 0 otherwise 
    y1=np.empty_like(find1)
    y2=np.empty_like(find2)
    neg1 = (rnd1(find1, 10, y1) >= 0)*1  
    neg2 = (rnd1(find2, 10, y2) >= 0)*1  
                
    # Make sure that no double relations (same relation between two words) are considered, 
    # otherwise ignore the second same relation. We don't need repetitions
    relations = []
    for i in range(len(indmax1)):
        relations.append((i,indmax1[i]))

    for j in range(len(indmax2)):
        if (indmax2[j],j) in relations:       
            neg2[j] = 0    
                
    sim = cross_similarity(M, w_1, w_2, eps, P, neg1, neg2, indmax1, indmax2)
    
    return sim

