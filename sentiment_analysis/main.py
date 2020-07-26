import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sentiment_analysis.data_preprocessing import load_data, tokenize
from sentiment_analysis.data_preprocessing import create_tfid_weighted_vec
from sentiment_analysis.train import train_classifier
from sentiment_analysis.word_vectors import train_w2v


def main(n_dim, nrows=1000):
    train_data = load_data('train.ft.txt', nrows=nrows)
    x_train = tokenize(train_data)
    y_train = np.array(train_data.sentiment)
    test_data = load_data('test.ft.txt', nrows=1000)
    x_test = tokenize(test_data)
    y_test = np.array(test_data.sentiment)
    
    w2v = train_w2v(x_train, n_dim=n_dim)
    
    vectorizer = TfidfVectorizer(analyzer=lambda x: x)
    vectorizer.fit(x_train)
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print('vocab size :', len(tfidf))

    train_vecs_w2v = create_tfid_weighted_vec(x_train, w2v, n_dim, tfidf)
    test_vecs_w2v = create_tfid_weighted_vec(x_test, w2v, n_dim, tfidf)
    
    scaler = StandardScaler()
    scaler.fit(train_vecs_w2v)
    
    train_vecs_w2v = scaler.transform(train_vecs_w2v)
    test_vecs_w2v = scaler.transform(test_vecs_w2v)
    
    model = train_classifier(train_vecs_w2v, y_train, n_dim)
    score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
    print('score {}'.format(score[1])) 
    
    models_path = os.path.join('..', 'models')
    
    tfidf_path = os.path.join(models_path, 'tfidf.pkl')
    pickle.dump(tfidf, open(tfidf_path, 'wb'))

    scaler_path = os.path.join(models_path, 'scaler.pkl')
    pickle.dump(scaler, open(scaler_path, 'wb'))
    
    w2v_path = os.path.join(models_path, 'w2v.model')
    w2v.save(w2v_path)
    
    model_path = os.path.join(models_path, 'model.h5')
    model.save(model_path)

if __name__ == '__main__':
    n_dim = 100
    main(n_dim, 200000)