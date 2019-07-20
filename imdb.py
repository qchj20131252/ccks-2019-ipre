'''
#Trains a Bidirectional LSTM on the IMDB sentiment classification task.
Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint
import model


max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 500
batch_size = 500

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)
print(y_train.shape)
print(y_train)


model = model.creat_model(max_features, maxlen)

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
mode='max')
callbacks_list = [checkpoint]

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          nb_epoch=10,
          validation_split=0.33,
          callbacks=callbacks_list,
          verbose=0)

print('Test...')

model = model.creat_model(max_features, maxlen)

model.load_weights("weights.best.hdf5")

scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))