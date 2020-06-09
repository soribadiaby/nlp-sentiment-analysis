from sentiment_analysis.data_preprocessing import (load_data, process_data,
                                                   create_tfid_weighted_vec)
from sentiment_analysis.train import train_classifier
from sentiment_analysis.word_vectors import train_w2v

def test_train_classifier():
    n_dim = 6
    data = load_data('amazon_cells_labelled.txt', nrows=10)
    x_train, x_test, y_train, y_test = process_data(data)
    w2v = train_w2v(x_train, n_dim=n_dim)
    train_vecs_w2v, test_vecs_w2v = create_tfid_weighted_vec(x_train,
                                                             x_test,
                                                             w2v, n_dim)
    
    model = train_classifier(train_vecs_w2v, y_train, n_dim, verb=False)
    score = model.evaluate(test_vecs_w2v, y_test)
    assert model.input_shape[1] == n_dim
    assert 0 <= score[1] <= 1
    