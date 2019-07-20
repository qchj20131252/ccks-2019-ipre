from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

def creat_model(max_features, maxlen):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model