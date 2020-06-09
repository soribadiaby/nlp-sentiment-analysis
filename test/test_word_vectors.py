from sklearn.feature_extraction.text import TfidfVectorizer
from sentiment_analysis.data_preprocessing import load_data, process_data
from sentiment_analysis.word_vectors import train_w2v, build_doc_vector

def test_train_w2v():
    n_dim = 7
    data = load_data('amazon_cells_labelled.txt', nrows=10)
    x_train, x_test, y_train, y_test = process_data(data)
    
    w2v = train_w2v(x_train, n_dim=n_dim)
    assert w2v.wv.vectors.shape[1] == n_dim
    

def test_build_doc_vector():
    n_dim = 5
    data = load_data('amazon_cells_labelled.txt', nrows=10)
    x_train, x_test, y_train, y_test = process_data(data)
    
    w2v = train_w2v(x_train, n_dim=n_dim)
    vectorizer = TfidfVectorizer(analyzer=lambda x: x)
    vectorizer.fit(x_train)
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    vectors = [build_doc_vector(doc, n_dim, w2v, tfidf) 
            for doc in x_train]
    
    assert all([len(vector[0]) == n_dim for vector in vectors])