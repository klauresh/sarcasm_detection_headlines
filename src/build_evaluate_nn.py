from sklearn.externals import joblib
from keras.models import Sequential, load_model
from keras.layers import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py

def build_nn(list_of_layers):
  '''Build neural network from a given list of layers.'''
  model = Sequential()
  model.add(Embedding(emb_matrix.shape[0],
                      emb_matrix.shape[1], 
                      input_length=max_len,
                      weights = [emb_matrix], 
                      trainable = False))
  for layer in list_of_layers:
    model.add(layer)
  model.summary() 
  
  model.compile(loss="binary_crossentropy",
               optimizer="adam", 
               metrics=["binary_accuracy"])
  return model



def evaluate_nn(model, X_train, X_val, y_train, y_val, filename, batch_size=32, early_stopping=False):
  '''Fit and evaluate neural network on validation data.'''
  take_best_model = ModelCheckpoint(str(filename)+".h5py", save_best_only=True)
  
  if early_stopping == True:
    early_stopping = EarlyStopping(patience=10, monitor="val_loss")
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2,
              batch_size=batch_size, 
              callbacks=[early_stopping, take_best_model])
   
  else:
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2,
              batch_size=batch_size, 
              callbacks=[take_best_model])
  
  joblib.dump(history, filename)

  model.load_weights(str(filename)+".h5py")
  return model.evaluate(X_val, y_val)[1]