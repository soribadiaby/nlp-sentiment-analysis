import numpy as np
from gensim.models.word2vec import Word2Vec


def train_w2v(x_train, n_dim):
    """
    Train word2vec
    
    Parameters
    ----------
    x_train : np.array
        training data where each line corresponds to a document
        
    n_dim : int
        dimensionality of our word vectors
        
    Returns
    -------
    w2v : gensim.Word2Vec
        trained word2vec model
    """
    w2v = Word2Vec(size=n_dim) 
    print('len words {}'.format(len(x_train)))
    w2v.build_vocab(x_train)
    w2v.train(x_train, total_examples=w2v.corpus_count, epochs=w2v.epochs)
    
    return w2v


def build_doc_vector(document, n_dim, w2v, tfidf):
    """
    Create a weighted sum with the words in the document where the weight are 
    tf-idf scores
    
    Parameters
    ----------
    document : list
        tokenized sample
        
    n_dim : int
        dimensionality of our word vectors
        
    w2v : gensim.Word2Vec
        word2vec model
        
    tfidf : dict
        vocabulary of words with their corresponding tf-idf score
        
    Returns
    -------
    vec : np.array
        vector that summarizes the document
    """
    vec = np.zeros((1, n_dim))
    for word in document:
        try:
            vec += w2v.wv[word] * tfidf[word]
        
        except:
            continue
            
    return vec