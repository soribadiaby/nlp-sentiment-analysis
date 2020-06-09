from sentiment_analysis.data_preprocessing import load_data, process_data
from sentiment_analysis.data_preprocessing import create_tfid_weighted_vec
from sentiment_analysis.train import train_classifier
from sentiment_analysis.word_vectors import train_w2v


def main(filename, n_dim):
    data = load_data(filename)
    x_train, x_test, y_train, y_test = process_data(data)
    w2v = train_w2v(x_train, n_dim=n_dim)
    train_vecs_w2v, test_vecs_w2v = create_tfid_weighted_vec(x_train,
                                                             x_test,
                                                             w2v, n_dim)
    print(train_vecs_w2v.shape)
    model = train_classifier(train_vecs_w2v, y_train, n_dim)
    score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
    print('score {}'.format(score[1])) 

if __name__ == '__main__':
    filename = 'train.ft.txt'
    n_dim = 100
    main(filename, n_dim)