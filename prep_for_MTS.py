#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize(corpus):
    """
    This function will return co-occurrence matrix, pairwise distances of all words in the corpus,
    {key: index, value: vocabulary} and {key: vocabulary, value: index}
    """
    count_vect = CountVectorizer()
    vect_corpus = count_vect.fit_transform(corpus)

    vocab = count_vect.get_feature_names()
    # Co-occurrence matrix
    cooc = vect_corpus.T.dot(vect_corpus).astype(np.uint32)

    D = pairwise_distances(cooc, metric = 'cosine', n_jobs = 5)

    Vocab = {}
    for index in range(len(vocab)):
        Vocab[vocab[index]] = index

    W = {}
    for key, value in Vocab.items():
        W[value] = key

    return [cooc, D, Vocab, W]
    
    
    
def w2v_distance_matrix(model, W): 
    """
    This function returns normalized cosine distance values for word2vec vectors
    """
    vectors_w2v = np.zeros((len(W),300))

    # Vector matrix with words order as in vocabulary
    for k in range(len(W)-1):
        vectors_w2v[k+1] = model.__getitem__(W[k+1])

    sim_w2v = cosine_similarity(vectors_w2v)
    dis_w2v = 1 - sim_w2v
    
    # Normalized distance
    d_norm_w2v = np.zeros((dis_w2v.shape[0], dis_w2v.shape[1]))
    for d in range(len(dis_w2v)):
        d_norm_w2v[d] = dis_w2v[d]/dis_w2v[d].max()
    
    return d_norm_w2v




def weights_tfidf(docs, indexed):
    """
    Assigns to each word in the text a value that was calculated in the parameter "indexed"
    """
    w_tfidf = []
    for i in range(len(docs)):
        doc = docs[i]
        w = np.zeros(len(doc))
        for j in range(len(doc)):
            w[j] = indexed[i].get(doc[j])
        w_tfidf.append(w)
        
    return w_tfidf



def norm_tfidf_w(corpus):
    """
    Calculates a tfidf values for each word in the document and returns a normalized values
    """
    corp_sep = [t.split() for t in corpus]

    tfidf_vec = TfidfVectorizer()
    transformed = tfidf_vec.fit_transform(raw_documents = corpus)
    index_value = {i[1]:i[0] for i in tfidf_vec.vocabulary_.items()}

    fully_indexed = []
    for row in transformed:
        fully_indexed.append({index_value[column]:value for (column,value) in zip(row.indices,row.data)})

    w_tfidf = weights_tfidf(corp_sep, fully_indexed)

    wn_tfidf = [w/max(w) for w in w_tfidf]

    return wn_tfidf

