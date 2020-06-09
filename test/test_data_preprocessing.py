import pandas as pd
from sentiment_analysis.data_preprocessing import (load_data, process_data, 
                                                   create_tfid_weighted_vec)
from sentiment_analysis.word_vectors import train_w2v

filenames = ['amazon_cells_labelled.txt', 'imdb_labelled.txt',
             'yelp_labelled.txt', 'train.ft.txt']

def test_load_data():
    for filename in filenames:
        data = load_data(filename, nrows=10)
        assert type(data) == pd.DataFrame
        assert 'text' in data.columns
        assert 'sentiment' in data.columns
        
        
def test_process_data():
    data = load_data('amazon_cells_labelled.txt', nrows=10)
    x_train, x_test, y_train, y_test = process_data(data)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    
    for tokens in x_train:
        assert all([type(e) == str for e in tokens])
        
    for tokens in x_test:
        assert all([type(e) == str for e in tokens])
        
        
def test_create_tfid_weighted_vec():
    n_dim = 3
    data = load_data('amazon_cells_labelled.txt', nrows=10)
    x_train, x_test, y_train, y_test = process_data(data)
    w2v = train_w2v(x_train, n_dim=n_dim)
    train_vecs_w2v, test_vecs_w2v = create_tfid_weighted_vec(x_train,
                                                             x_test,
                                                             w2v, 
                                                             n_dim=n_dim)
    assert train_vecs_w2v.shape == (len(x_train), n_dim)
    assert test_vecs_w2v.shape == (len(x_test), n_dim)