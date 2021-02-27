import json
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

with open('static/intent.json','rb') as file:
	df = json.load(file)

training = []
training_labels = []
labels = []
responses = []

for intent in df['intents']:
	for pattern in intent['patterns']:
		training.append(pattern)
		training_labels.append(intent['tag'])
	responses.append(intent['responses'])

	if intent['tag'] not in labels:
		labels.append(intent['tag'])

num = len(labels)

encoder = LabelEncoder()
encoder.fit(training_labels)
training_labels = encoder.transform(training_labels)

vocab_size = 1000
embed_dim = 16
max_len = 20
oov_taken = "oov"

token = Tokenizer(num_words=vocab_size,oov_token=oov_taken)
token.fit_on_texts(training)
word_index = token.word_index
sequences = token.texts_to_sequences(training)
pad_sequence = pad_sequences(sequences, truncating='post', maxlen=max_len)

model = Sequential()
model.add(Embedding(vocab_size,embed_dim,input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

modal = model.fit(pad_sequence, np.array(training_labels), epochs=500)

model.save('chat')

with open('static/token.pickle','wb') as f:
	pickle.dump(token,f,protocol=pickle.HIGHEST_PROTOCOL)

with open('static/encoder.pickle','wb') as e:
	pickle.dump(encoder,e,protocol=pickle.HIGHEST_PROTOCOL)