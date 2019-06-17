from sklearn.externals import joblib
import numpy as np


def load_embeddings(path):
    '''Function to load pre-trained embeddings from pickled file.'''
    with open(path,'rb') as f:
        emb_arr = joblib.load(f)
    return emb_arr
  
def build_matrix(word_index, path):
    '''Function to bild matrix with embeddings for words in our vocabulary'''
    emb_index = load_embeddings(path)
    emb_matrix = np.zeros((len(word_index), 300))
    unknown_words = []
    
    for w, i in word_index.items():
      try:
        vect = emb_index[w]
        if vect is not None:
          emb_matrix[i] = vect
      except:
        unknown_words.append(w)
    return emb_matrix, unknown_words
