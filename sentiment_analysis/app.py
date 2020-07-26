import os
import pickle
from flask import Flask, request, render_template
from gensim.models import Word2Vec
from keras.models import load_model
from nltk.tokenize import word_tokenize
import tensorflow as tf
import numpy as np
from sentiment_analysis.data_preprocessing import filter_tokens, stem
from sentiment_analysis.word_vectors import build_doc_vector

graph = tf.get_default_graph()
app = Flask(__name__)

models_path = os.path.join('..', 'models')
model_path = os.path.join(models_path, 'model.h5')
tfidf_path = os.path.join(models_path, 'tfidf.pkl')


scaler_path = os.path.join(models_path, 'scaler.pkl')
w2v_path = os.path.join('..', 'models', 'w2v.model')

model = load_model(model_path)

tfidf = pickle.load(open(tfidf_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))


w2v = Word2Vec.load(w2v_path)
n_dim = w2v.vector_size
    
@app.route("/")
def index():
    return render_template('index.html', pred = 0)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['review']
    if not data:
        prediction_text = "Please write a non-empty review"
        
    else:
        tokens = word_tokenize(data)
        tokens = filter_tokens(tokens)
        tokens = stem(tokens)
        tokens = np.array(tokens)
        doc_vector = build_doc_vector(tokens, n_dim, w2v, tfidf)
        doc_vector = scaler.transform(doc_vector)
        with graph.as_default():
            prediction = (model.predict(doc_vector)[0][0] > 0.5) * 1
            
        if prediction == 0:
            prediction_str = "negative"
            
        elif prediction == 1:
            prediction_str = "positive"
            
        prediction_text = f"This review is {prediction_str}"
    return render_template('index.html', pred=prediction_text)


def main():
    """Run the app."""
    app.run(debug=True)  


if __name__ == '__main__':
    main()