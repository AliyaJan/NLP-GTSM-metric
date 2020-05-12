#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def w2v_corpus(corpus_sep):
    """
    corpus_sep: ex of corpus_sep[0] = ['preliminary', 'report', 'international', 'algebraic', 'language']
    """
    modelGoogle = w2v.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    modelGoogle.init_sims(replace=True)

    # Delete elements that are not strings
    for i in range(len(corpus_sep)):
        corpus_sep[i] = [j for j in corpus_sep[i] if type(j)==np.str_]

    # Clean corpus from words that are not in word2vec
    w2v_corpus = []
    for publ in tqdm(corpus_sep):
        new_words = list(filter(lambda w: w in modelGoogle.vocab, publ))
        w2v_corpus.append(new_words)
    
    return [modelGoogle, w2v_corpus]
    
    
    
    
def glove_corpus(corpus_sep):  
    """
    corpus_sep: ex of corpus_sep[0] = ['preliminary', 'report', 'international', 'algebraic', 'language']
    """
    embeddings_dict = {}
    with open("glove.6B.300d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

    glove_corpus = []

    for doc in tqdm(corpus_sep):
        glove_sent = []
        for sent in doc:
            new_words = list(filter(lambda w: w in embeddings_dict, sent))
            glove_sent.append(new_words)
        glove_corpus.append(glove_sent)
        
    return [embeddings_dict, glove_corpus]




def bert(corpus):
    
    import torch
    from pytorch_pretrained_bert import BertTokenizer, BertModel
    

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    
    
    def word_sent_emb(corpus):

    sentence_embedding = []
    sent_vecs_sum = []
    word_emb = []
    new_corpus = []

    for text in corpus:

        # Add the special tokens.
        marked_text = "[CLS] " + text + " [SEP]"
        
        # Split the sentence into tokens.
        tokenized_text = tokenizer.tokenize(marked_text)

        # If large than 512 tokens, leave only first and last 256 tokens in the text
        if len(tokenized_text) >= 512:
            end = int(len(tokenized_text)-256)
            del tokenized_text[256:end]

        # Map the token strings to their vocabulary indeces.
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        
        # Mark each of the tokens as belonging to sentence "1".
        segments_ids = [1] * len(indexed_tokens)
        
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        
        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = model(tokens_tensor, segments_tensors)
        
        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(encoded_layers, dim=0)

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)

        token_vecs_sum = []

        # For each token in the sentence...
        for token in token_embeddings:

            # `token` is a [.. x 768] tensor
            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)
    #     sent_vecs_sum.append(token_vecs_sum)

        # Recreate the new BERT corpus after tokenizing (words can be also tokenized if not found in bert vocab), 
        # deleting tokens that are more than 512 
        d = {} 
        words = {}
        separator = 1
        for t in range(len(tokenized_text)):
            d[t] = token_vecs_sum[t]
            words[t] = tokenized_text[t]
            if '##' in tokenized_text[t]:
                d[t-separator] = (token_vecs_sum[t] + d[t-separator])/2
                words[t-separator] = words[t-separator] + tokenized_text[t]
                words[t-separator] = ''.join(filter(str.isalpha,  words[t-separator]))
                del d[t]
                del words[t]
                separator += 1
            else:
                separator = 1   
        word_emb.append(list(d.values())[1:-1])      # excluding CLS and SEP
        new_corpus.append(list(words.values())[1:-1])

    
    return [word_emb, new_corpus]
    

