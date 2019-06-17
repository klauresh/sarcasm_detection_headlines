import matplotlib.pyplot as plt
import os
from sklearn.externals import joblib

def plot_accuracy(path, models):
  '''Function to plot accuracy on train and validation sets.'''
  for model in models:
    history = joblib.load(os.path.join(path_to_results, model))
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title(f"model {model} accuracy")
    plt.ylabel('accuracy')
    plt.ylim(0.6, 1)
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

def plot_loss(path, models):
  '''Function to plot loss on train and validation sets.'''
  for model in models:
    history = joblib.load(os.path.join(path_to_results, model))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f"model {model} loss")
    plt.ylabel('loss')
    #plt.ylim(0.6, 1)
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
