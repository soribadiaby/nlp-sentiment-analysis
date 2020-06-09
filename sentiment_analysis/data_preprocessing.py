import os
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentiment_analysis.word_vectors import build_doc_vector


def load_data(filename, nrows=10000):
    """
    Load the dataset
    
    Parameters
    ----------
    filename : str
        the file name
        
    Returns
    -------
    train_data : pd.DataFrame
        the training data (with 'text' and 'sentiment' columns)
    """
    if filename in ['amazon_cells_labelled.txt', 'imdb_labelled.txt', 
                    'yelp_labelled.txt']:
        data_path = os.path.join('..', 'data', filename)
        data = pd.read_csv(data_path, sep='\t', header=None, nrows=nrows,
                           names=['text', 'sentiment'])
        
    elif filename in ['test.ft.txt', 'train.ft.txt']:
        data_path = os.path.join('..', 'data', filename)
        data = pd.read_csv(data_path, sep='^([^ ]*)', header=None,
                           engine='python', nrows=nrows,
                           names=[0, 'sentiment', 'text'])
        data.drop(columns=0, inplace=True)
        data = data.replace({'__label__2':1, '__label__1':0})
    
    return data
    

def process_data(data):
    """
    Create train and test data from initial dataframe
    
    Parameters
    ----------
    data : pd.DataFrame
        dataset with 'text' and 'sentiment' columns 
        
    Returns
    -------
    splitting : tuple
        train-test split of input
    """
    tokens = data['text'].apply(lambda x: word_tokenize(x.lower())) 
    tokens = np.array(tokens)
    sentiment = np.array(data.sentiment)
    x_train, x_test, y_train, y_test = train_test_split(tokens,
                                                        sentiment,
                                                        test_size=0.25)
    
    return x_train, x_test, y_train, y_test


def create_tfid_weighted_vec(x_train, x_test, w2v, n_dim):
    """
    Create train, test vecs using the tf-idf weighting method
    
    Parameters
    ----------
    x_train : np.array
        training data (tokenized) where each line corresponds to a document
        
    x_test : np.array
        testing data (tokenized) where each line corresponds to a document
        
    w2v : gensim.Word2Vec
        word2vec model
    
    n_dim : int
        dimensionality of our word vectors
        
    Returns
    -------
    train_test : tuple
        data ready for the model, shape (n_samples, n_dim)    
    """
    vectorizer = TfidfVectorizer(analyzer=lambda x: x)
    vectorizer.fit(x_train)
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print('vocab size :', len(tfidf))
    
    train_vecs_w2v = np.concatenate(
            [build_doc_vector(doc, n_dim, w2v, tfidf) 
            for doc in x_train])
    scaler = StandardScaler()
    scaler.fit(train_vecs_w2v)
    train_vecs_w2v = scaler.transform(train_vecs_w2v)
    
    test_vecs_w2v = np.concatenate(
            [build_doc_vector(doc, n_dim, w2v, tfidf) 
            for doc in x_test])
    test_vecs_w2v = scaler.transform(test_vecs_w2v)
    
    return train_vecs_w2v, test_vecs_w2v