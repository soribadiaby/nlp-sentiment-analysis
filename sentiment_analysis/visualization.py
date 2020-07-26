# importing bokeh library for interactive dataviz
import bokeh.plotting as bp
from bokeh.models import HoverTool
from bokeh.plotting import show, output_notebook
import gensim
import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
from sklearn.manifold import TSNE
from wordcloud import WordCloud, STOPWORDS


def plot_word_vectors(w2v):
    """
    Scatter plot of word vectors
    
    Parameters
    ----------
    w2v : gensim.Word2Vec
        word2vec model
    """
    output_notebook()
    plot_tools = "pan,wheel_zoom,box_zoom,reset,hover,previewsave"
    plot_tfidf = bp.figure(plot_width=700,
                           plot_height=600,
                           title="A map of the word vectors",
                           tools=plot_tools,
                           x_axis_type=None,
                           y_axis_type=None,
                           min_border=1)
    
    word_vectors = [w2v.wv[w] for w in w2v.wv.vocab.keys()] 
    
    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    tsne_w2v = tsne.fit_transform(word_vectors)
    
    tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
    tsne_df['words'] = w2v.wv.vocab.keys()
    
    plot_tfidf.scatter(x='x', y='y', source=tsne_df)
    hover = plot_tfidf.select(dict(type=HoverTool))
    hover.tooltips={"word": "@words"}
    show(plot_tfidf)
    

def show_wordcloud(data, title = None):
    """
    Word cloud
    
    Parameters
    ----------
    data : list
        list of (string) documents 
    """
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
    
    
def show_topics(corpus):
    """
    Topics visualization
    
    Parameters
    ----------
    corpus : list
        corpus of (string) documents
    """
    dic=gensim.corpora.Dictionary(corpus)
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]
    lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 4, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)

    lda_vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
    pyLDAvis.enable_notebook()
    pyLDAvis.display(lda_vis)