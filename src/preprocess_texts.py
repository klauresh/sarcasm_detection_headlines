import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

def preprocess_texts(X):
  # 1.step: turning all words into lower case
  X_1 = X.str.lower()

  # 2.step: tokenizing texts (getting a list of all words used)
  X_2 = X.apply(nltk.word_tokenize)

  # 3.step: removing the stopwords
  stopwords = nltk.corpus.stopwords.words("english")
  X_3 = [[word for word in headline if word not in stopwords] for headline in X_2]

  # 4.step: stemming the words
  stemmer = nltk.PorterStemmer()
  X_4 = [[stemmer.stem(word) for word in headline] for headline in X_3]
	  
  # 6.step: getting rid of punctuation  
  X_5 = [" ".join(word) for word in X_4]
  table = str.maketrans({key: None for key in string.punctuation})
  X_6 = [w.translate(table) for w in X_5]
	  
  return X_6
