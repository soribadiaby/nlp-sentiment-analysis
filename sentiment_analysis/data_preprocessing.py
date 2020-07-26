import os
import numpy as np
import pandas as pd
import swifter
from nltk.tokenize import word_tokenize
from sentiment_analysis.word_vectors import build_doc_vector
from nltk.stem import PorterStemmer


def load_data(filename, nrows=None):
    """
    Load the dataset
    
    Parameters
    ----------
    filename : str
        the file name (train.ft.txt or test.ft.txt)
        
    Returns
    -------
    data : pd.DataFrame
         data (with 'text' and 'sentiment' columns)
    """
    data_path = os.path.join('..', 'data', filename)
    data = pd.read_csv(data_path, sep='^([^ ]*)', header=None,
                       engine='python', nrows=nrows,
                       names=[0, 'sentiment', 'text'])
    data.drop(columns=0, inplace=True)
    data = data.replace({'__label__2':1, '__label__1':0})
    
    return data

def filter_tokens(tokens_list):
    return [word for word in tokens_list if word.isalpha() and len(word) > 2]


def stem(tokens_list):
    ps = PorterStemmer()
    return [ps.stem(word) for word in tokens_list]

    
def tokenize(data):
    tokens = (data['text']
    .swifter.apply(lambda x: word_tokenize(x.lower()))
    .apply(filter_tokens)
    ) 
    tokens = np.array(tokens)
    
    return tokens


def create_tfid_weighted_vec(tokens, w2v, n_dim, tfidf):
    """
    Create train, test vecs using the tf-idf weighting method
    
    Parameters
    ----------
    tokens : np.array
        data (tokenized) where each line corresponds to a document

    w2v : gensim.Word2Vec
        word2vec model
    
    n_dim : int
        dimensionality of our word vectors
        
    Returns
    -------
    vecs_w2v : np.array
        data ready for the model, shape (n_samples, n_dim)    
    """
    vecs_w2v = np.concatenate(
            [build_doc_vector(doc, n_dim, w2v, tfidf) 
            for doc in tokens])    
    
    return vecs_w2v