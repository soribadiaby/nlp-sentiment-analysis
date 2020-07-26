from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization


def get_baseline_model(n_dim):
    """
    Baseline model
    
    Parameters
    ----------
    n_dim : int
        dimensionality of our word vectors
    """
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=n_dim))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model