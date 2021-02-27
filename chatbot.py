import pickle
import json
from tensorflow import keras
import colorama
from colorama import Fore, Style, Back
import numpy as np

colorama.init()

with open('static/intent.json','rb') as file:
	df = json.load(file)

def chat():
	model = keras.models.load_model('chat')

	with open('static/token.pickle','rb') as t:
		token = pickle.load(t)

	with open('static/encoder.pickle','rb') as e:
		encoder = pickle.load(e)

	max_len = 20

	while True:
		print(Fore.LIGHTBLUE_EX + "User" + Style.RESET_ALL, end=" ")
		inp = input()

		if inp.lower() == "quit":
			break

		result = model.predict(keras.preprocessing.sequence.pad_sequences(token.texts_to_sequences([inp]),
			truncating='post',maxlen=max_len))
		tag = encoder.inverse_transform([np.argmax(result)])

		for i in df['intents']:
			if i['tag'] == tag:
				print(Fore.GREEN + 'Chatbot' + Style.RESET_ALL, np.random.choice(i['responses']))

print(Fore.YELLOW + 'Start messaging with the bot (type quit to stop)!' +Style.RESET_ALL)
chat()