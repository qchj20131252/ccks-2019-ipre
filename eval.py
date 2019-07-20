import numpy as np

from keras.preprocessing import sequence
from keras.datasets import imdb
import model

max_features = 20000
maxlen = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

y_test = np.array(y_test)

model = model.creat_model(max_features, maxlen)
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

model.load_weights("weights.best.hdf5")

scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
