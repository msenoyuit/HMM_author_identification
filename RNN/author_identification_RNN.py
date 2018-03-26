import sys
from matplotlib import pyplot
from keras.preprocessing import text
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LSTM, Embedding, GRU
import RNN_preprocessing as RPP

seq_size = 100
sample_length =500
char_level=True
batch_size = 256
epochs = 5
verbose = 1

x_seq, y_seq, plays_token, authors = RPP.getListOfSequences(char_level=char_level,seq_size=seq_size, verbose=verbose)
print(x_seq.shape)
vocab_size = len(plays_token.word_index) +1

if verbose:
    print('Building Model')
model = Sequential()
model.add(Embedding(vocab_size, 500, mask_zero=True, input_length=seq_size))
model.add(LSTM(500))
model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',  optimizer='adam', metrics=['sparse_categorical_accuracy'])


history = model.fit(x_seq, y_seq, verbose=1, epochs=epochs, batch_size=batch_size)
score = model.evaluate(x_seq, y_seq, batch_size=128)

print(score)
