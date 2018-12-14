'''
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from keras.initializers import Constant
from keras.layers import (Conv1D, Dense, Dropout, Embedding,
                          GlobalMaxPooling1D, Input, MaxPooling1D,
                          regularizers)
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model, to_categorical

start = time.clock()

# $HOME/.keras/keras.json
# "backend": "tensorflow"

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000  # 最大序列长度
MAX_NUM_WORDS = 20000  # 最大单词数量
EMBEDDING_DIM = 100  # 嵌入维度
VALIDATION_SPLIT = 0.2

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

print('Processing text dataset')

# 20_sewsgroup
texts = []
labels_index = {}
labels = []
for name in sorted(os.listdir(TEXT_DATA_DIR)):  # 一级目录
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):  # 二级目录
        label_id = len(labels_index)
        labels_index[name] = label_id  # name->[0,19]
        for fname in sorted(os.listdir(path)):  # 文件
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3, ) else {
                    'encoding': 'latin-1'
                }
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n')
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(label_id)

print('Found %s texts.' % len(texts))

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)  # (19997,1000)
print('Shape of label tensor:', labels.shape)  # (19997,20)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

# Glove
print('Preparing embedding matrix.')

num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():  # i [1,20000]
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    embeddings_initializer=Constant(embedding_matrix),
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False)

print('Training model.')

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Dropout(0.5)(embedded_sequences)
x = Conv1D(
    128, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = MaxPooling1D(5)(x)
x = Conv1D(
    128, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = MaxPooling1D(5)(x)
x = Conv1D(
    128, 5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)

model.compile(
    loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=100,
    validation_data=(x_val, y_val))

end = time.clock()
print('Running time: %s Seconds' % (end - start))

print(model.summary())
plot_model(model, show_shapes=True, to_file='model.png')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Traning acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure

plt.plot(epochs, loss, 'bo', label='Traning loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
