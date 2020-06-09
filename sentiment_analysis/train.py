from sentiment_analysis.models import get_baseline_model

        
def train_classifier(train_vecs_w2v, y_train, n_dim, verb=2):
    model = get_baseline_model(n_dim=n_dim)   
    model.fit(train_vecs_w2v, y_train, epochs=20, batch_size=32, verbose=verb)
    
    return model
        